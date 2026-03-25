//! Real-time GPU ray tracer viewer using Vulkan compute.
//!
//! Renders the verified ray tracer scene (4 spheres + ground plane) at
//! interactive frame rates using a GLSL compute shader that mirrors the
//! verified Verus specs (ray_hits_sphere_nodiv, ray_hits_plane,
//! lambertian_clamped, camera_ray).
//!
//! Controls:
//!   WASD      — fly forward/left/back/right
//!   Space     — fly up
//!   LShift    — fly down
//!   LCtrl     — speed boost (5x)
//!   Mouse     — look around (click to capture, Escape to release)
//!   +/=  -/_  — zoom in/out (FOV)
//!
//! Run: `cargo run --bin raytracer_viewer --features viewer`

use std::collections::HashSet;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId, WindowAttributes},
};

// ═══════════════════════════════════════════════════════════════════════════
// Vulkan backend
// ═══════════════════════════════════════════════════════════════════════════

mod vulkan {
    use super::*;
    use std::sync::Arc;
    use std::time::Instant;
    use ash::vk;
    use ash::vk::Handle;
    use builtin::Ghost;
    use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
    use verus_vulkan::vk_context::VulkanContext;
    use verus_vulkan::ffi;
    use verus_vulkan::runtime::surface::RuntimeSurface;
    use verus_vulkan::runtime::swapchain::RuntimeSwapchain;
    use verus_vulkan::runtime::shader_module::RuntimeShaderModule;
    use verus_vulkan::runtime::command_pool::RuntimeCommandPool;
    use verus_vulkan::runtime::command_buffer::RuntimeCommandBuffer;
    use verus_vulkan::runtime::fence::RuntimeFence;
    use verus_vulkan::runtime::semaphore::RuntimeSemaphore;
    use verus_vulkan::runtime::device::RuntimeDevice;
    use verus_vulkan::runtime::queue::RuntimeQueue;
    use verus_vulkan::runtime::framebuffer::RuntimeImageView;
    use verus_vulkan::runtime::pipeline::RuntimeComputePipeline;
    use verus_vulkan::runtime::descriptor::{RuntimeDescriptorSetLayout, RuntimeDescriptorPool, RuntimeDescriptorSet};

    #[repr(C)]
    struct PushConstants {
        eye: [f32; 3],      fov: f32,
        forward: [f32; 3],  width: u32,
        right: [f32; 3],    height: u32,
        up: [f32; 3],       time: f32,
    }

    pub struct VkState {
        ctx: VulkanContext,
        _dev: RuntimeDevice,
        raw_surface: vk::SurfaceKHR,
        _surface: RuntimeSurface,
        queue: RuntimeQueue,
        swapchain: RuntimeSwapchain,
        swapchain_images: Vec<u64>,
        image_views: Vec<RuntimeImageView>,
        compute_pipeline: RuntimeComputePipeline,
        pipeline_layout_handle: u64,
        descriptor_set_layout: RuntimeDescriptorSetLayout,
        descriptor_pool: RuntimeDescriptorPool,
        descriptor_sets: Vec<RuntimeDescriptorSet>,
        shader_module: RuntimeShaderModule,
        command_pool: RuntimeCommandPool,
        command_buffers: Vec<RuntimeCommandBuffer>,
        image_available_sem: RuntimeSemaphore,
        render_finished_sem: RuntimeSemaphore,
        in_flight_fence: RuntimeFence,
        _format: vk::Format,
        width: u32,
        height: u32,
        // Camera
        pub eye: [f32; 3],
        pub yaw: f32,
        pub pitch: f32,
        pub fov: f32,
        pub move_speed: f32,
        // Input
        pub keys_held: HashSet<KeyCode>,
        pub last_frame_time: Instant,
        pub mouse_captured: bool,
        pub start_time: Instant,
    }

    impl VkState {
        pub fn new(window: Arc<Window>) -> Self {
            let size = window.inner_size();
            let width = size.width.max(1);
            let height = size.height.max(1);

            let display_handle = window.display_handle().unwrap();
            let surface_extensions =
                ash_window::enumerate_required_extensions(display_handle.as_raw())
                    .expect("Failed to get required surface extensions");

            let device_exts: Vec<*const i8> = vec![ash::khr::swapchain::NAME.as_ptr()];

            let ctx = unsafe {
                VulkanContext::new("raytracer_viewer", true, surface_extensions, &device_exts, 0)
            };

            let raw_surface = unsafe {
                ash_window::create_surface(
                    &ctx.entry,
                    &ctx.instance,
                    display_handle.as_raw(),
                    window.window_handle().unwrap().as_raw(),
                    None,
                )
            }
            .expect("Failed to create Vulkan surface");
            let surface = ffi::vk_create_surface(&ctx, Ghost::assume_new(), raw_surface.as_raw());

            let dev = ffi::vk_create_device(&ctx, Ghost::assume_new());
            let queue = ffi::vk_get_device_queue(&ctx, 0, 0, Ghost::assume_new());

            let surface_formats = unsafe {
                ctx.surface_loader
                    .get_physical_device_surface_formats(ctx.physical_device, raw_surface)
            }
            .expect("Failed to query surface formats");
            let format = surface_formats
                .iter()
                .find(|f| f.format == vk::Format::B8G8R8A8_UNORM)
                .unwrap_or(&surface_formats[0])
                .format;

            let image_count = 2u64;
            let swapchain = ffi::vk_create_swapchain(
                &ctx,
                Ghost::assume_new(),
                image_count,
                raw_surface.as_raw(),
                format.as_raw() as u32,
                width,
                height,
                vk::PresentModeKHR::FIFO.as_raw() as u32,
                (vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::COLOR_ATTACHMENT).as_raw(),
            );

            let swapchain_images = ffi::vk_get_swapchain_images(&ctx, &swapchain);
            let mut image_views = Vec::new();
            for &img in swapchain_images.iter() {
                let view = ffi::vk_create_image_view(
                    &ctx,
                    Ghost::assume_new(),
                    img,
                    format.as_raw() as u32,
                    vk::ImageAspectFlags::COLOR.as_raw() as u32,
                );
                image_views.push(view);
            }

            let descriptor_set_layout = ffi::vk_create_descriptor_set_layout_typed(
                &ctx,
                Ghost::assume_new(),
                &[(
                    0,
                    vk::DescriptorType::STORAGE_IMAGE.as_raw() as u32,
                    1,
                    vk::ShaderStageFlags::COMPUTE.as_raw(),
                )],
            );

            let pipeline_layout_handle = ffi::vk_create_pipeline_layout_push(
                &ctx,
                &[descriptor_set_layout.handle],
                vk::ShaderStageFlags::COMPUTE.as_raw(),
                0,
                std::mem::size_of::<PushConstants>() as u32,
            );

            let spv_code = spv_to_u32(include_bytes!("shaders/raytracer.comp.spv"));
            let shader_module = ffi::vk_create_shader_module(&ctx, Ghost::assume_new(), &spv_code);

            let compute_pipeline = ffi::vk_create_compute_pipeline(
                &ctx,
                Ghost::assume_new(),
                pipeline_layout_handle,
                shader_module.handle,
            );

            let mut descriptor_pool = ffi::vk_create_descriptor_pool_typed(
                &ctx,
                Ghost::assume_new(),
                image_count,
                &[(vk::DescriptorType::STORAGE_IMAGE.as_raw() as u32, image_count as u32)],
            );

            let mut descriptor_sets = Vec::new();
            for i in 0..image_count as usize {
                let mut ds = ffi::vk_allocate_descriptor_sets(
                    &ctx,
                    &mut descriptor_pool,
                    Ghost::assume_new(),
                    Ghost::assume_new(),
                    descriptor_set_layout.handle,
                );
                ffi::vk_update_descriptor_sets_image(
                    &ctx,
                    &mut ds,
                    Ghost::assume_new(),
                    Ghost::assume_new(),
                    0,
                    vk::DescriptorType::STORAGE_IMAGE.as_raw() as u32,
                    image_views[i].handle,
                    vk::ImageLayout::GENERAL.as_raw() as u32,
                );
                descriptor_sets.push(ds);
            }

            let command_pool = ffi::vk_create_command_pool(&ctx, Ghost::assume_new(), 0, true);
            let mut command_buffers = Vec::new();
            for _ in 0..image_count as usize {
                let cb = ffi::vk_allocate_command_buffer(
                    &ctx,
                    Ghost::assume_new(),
                    command_pool.handle,
                );
                command_buffers.push(cb);
            }

            let in_flight_fence = ffi::vk_create_fence(&ctx, Ghost::assume_new(), true);
            let image_available_sem = ffi::vk_create_semaphore(&ctx, Ghost::assume_new());
            let render_finished_sem = ffi::vk_create_semaphore(&ctx, Ghost::assume_new());

            // Camera: initial position looking at the scene
            let eye = [0.0f32, 3.0, -3.0];
            let target = [0.5f32, 1.0, 5.0];
            let dx = target[0] - eye[0];
            let dy = target[1] - eye[1];
            let dz = target[2] - eye[2];
            let yaw = dz.atan2(dx);
            let pitch = dy.atan2((dx * dx + dz * dz).sqrt());

            eprintln!(
                "Verified Ray Tracer Viewer: {}x{}, format {:?}",
                width, height, format
            );
            eprintln!("Controls: WASD=move, Mouse=look (click to capture, Esc=release)");
            eprintln!("          Space/LShift=up/down, LCtrl=boost, +/-=FOV");

            VkState {
                ctx,
                _dev: dev,
                raw_surface,
                _surface: surface,
                queue,
                swapchain,
                swapchain_images,
                image_views,
                compute_pipeline,
                pipeline_layout_handle,
                descriptor_set_layout,
                descriptor_pool,
                descriptor_sets,
                shader_module,
                command_pool,
                command_buffers,
                image_available_sem,
                render_finished_sem,
                in_flight_fence,
                _format: format,
                width,
                height,
                eye,
                yaw,
                pitch,
                fov: 0.8,
                move_speed: 2.0,
                keys_held: HashSet::new(),
                last_frame_time: Instant::now(),
                mouse_captured: false,
                start_time: Instant::now(),
            }
        }

        pub fn resize(&mut self, width: u32, height: u32) {
            if width == 0 || height == 0 {
                return;
            }
            self.width = width;
            self.height = height;
        }

        fn camera_vectors(&self) -> ([f32; 3], [f32; 3], [f32; 3]) {
            let (sp, cp) = (self.pitch.sin(), self.pitch.cos());
            let (sy, cy) = (self.yaw.sin(), self.yaw.cos());

            let forward = [cp * cy, sp, cp * sy];
            let rx = -forward[2];
            let rz = forward[0];
            let rlen = (rx * rx + rz * rz).sqrt();
            let right = if rlen > 1e-6 {
                [rx / rlen, 0.0, rz / rlen]
            } else {
                [1.0, 0.0, 0.0]
            };
            let up = [
                right[1] * forward[2] - right[2] * forward[1],
                right[2] * forward[0] - right[0] * forward[2],
                right[0] * forward[1] - right[1] * forward[0],
            ];

            (forward, right, up)
        }

        pub fn update_camera(&mut self) {
            let dt = self.last_frame_time.elapsed().as_secs_f32();
            self.last_frame_time = Instant::now();

            let speed = if self.keys_held.contains(&KeyCode::ControlLeft) {
                self.move_speed * 5.0
            } else {
                self.move_speed
            };

            let (forward, right, _up) = self.camera_vectors();

            if self.keys_held.contains(&KeyCode::KeyW) {
                for i in 0..3 { self.eye[i] += forward[i] * speed * dt; }
            }
            if self.keys_held.contains(&KeyCode::KeyS) {
                for i in 0..3 { self.eye[i] -= forward[i] * speed * dt; }
            }
            if self.keys_held.contains(&KeyCode::KeyA) {
                for i in 0..3 { self.eye[i] -= right[i] * speed * dt; }
            }
            if self.keys_held.contains(&KeyCode::KeyD) {
                for i in 0..3 { self.eye[i] += right[i] * speed * dt; }
            }
            if self.keys_held.contains(&KeyCode::Space) {
                self.eye[1] += speed * dt;
            }
            if self.keys_held.contains(&KeyCode::ShiftLeft) {
                self.eye[1] -= speed * dt;
            }
        }

        pub fn render(&mut self) {
            self.update_camera();

            let (forward, right, up) = self.camera_vectors();
            let time = self.start_time.elapsed().as_secs_f32();

            unsafe { let _ = self.ctx.device.device_wait_idle(); }

            ffi::vk_wait_for_fences(
                &self.ctx,
                &mut self.in_flight_fence,
                Ghost::assume_new(),
                u64::MAX,
            );
            ffi::vk_reset_fences(&self.ctx, &mut self.in_flight_fence);

            let idx = ffi::vk_acquire_next_image(
                &self.ctx,
                &mut self.swapchain,
                self.image_available_sem.handle,
                0,
                u64::MAX,
            );

            let cb = &mut self.command_buffers[idx as usize];
            let image_handle = self.swapchain_images[idx as usize];

            let pc = PushConstants {
                eye: self.eye,
                fov: self.fov,
                forward,
                width: self.width,
                right,
                height: self.height,
                up,
                time,
            };
            let pc_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    &pc as *const PushConstants as *const u8,
                    std::mem::size_of::<PushConstants>(),
                )
            };

            ffi::vk_begin_command_buffer(&self.ctx, cb);

            // UNDEFINED → GENERAL
            ffi::vk_cmd_pipeline_barrier_image(
                &self.ctx,
                cb,
                Ghost::assume_new(),
                vk::PipelineStageFlags::TOP_OF_PIPE.as_raw(),
                vk::PipelineStageFlags::COMPUTE_SHADER.as_raw(),
                &[(
                    image_handle,
                    vk::ImageLayout::UNDEFINED.as_raw() as u32,
                    vk::ImageLayout::GENERAL.as_raw() as u32,
                    vk::AccessFlags::empty().as_raw(),
                    vk::AccessFlags::SHADER_WRITE.as_raw(),
                )],
            );

            ffi::vk_cmd_bind_pipeline(
                &self.ctx,
                cb,
                vk::PipelineBindPoint::COMPUTE.as_raw() as u32,
                self.compute_pipeline.handle,
            );

            ffi::vk_cmd_bind_descriptor_sets(
                &self.ctx,
                cb,
                Ghost::assume_new(),
                vk::PipelineBindPoint::COMPUTE.as_raw() as u32,
                self.pipeline_layout_handle,
                0,
                &[self.descriptor_sets[idx as usize].handle],
            );

            ffi::ffi_cmd_push_constants(
                &self.ctx,
                cb.handle,
                self.pipeline_layout_handle,
                vk::ShaderStageFlags::COMPUTE.as_raw(),
                0,
                pc_bytes,
            );

            let group_x = (self.width + 15) / 16;
            let group_y = (self.height + 15) / 16;
            ffi::vk_cmd_dispatch(&self.ctx, cb, group_x, group_y, 1);

            // GENERAL → PRESENT_SRC_KHR
            ffi::vk_cmd_pipeline_barrier_image(
                &self.ctx,
                cb,
                Ghost::assume_new(),
                vk::PipelineStageFlags::COMPUTE_SHADER.as_raw(),
                vk::PipelineStageFlags::BOTTOM_OF_PIPE.as_raw(),
                &[(
                    image_handle,
                    vk::ImageLayout::GENERAL.as_raw() as u32,
                    vk::ImageLayout::PRESENT_SRC_KHR.as_raw() as u32,
                    vk::AccessFlags::SHADER_WRITE.as_raw(),
                    vk::AccessFlags::empty().as_raw(),
                )],
            );

            ffi::vk_end_command_buffer(&self.ctx, cb);

            let wait_stage = vk::PipelineStageFlags::COMPUTE_SHADER.as_raw();
            ffi::vk_queue_submit(
                &self.ctx,
                &self.queue,
                Ghost::assume_new(),
                Ghost::assume_new(),
                Ghost::assume_new(),
                &[cb.handle],
                &[self.image_available_sem.handle],
                &[wait_stage],
                &[self.render_finished_sem.handle],
                self.in_flight_fence.handle,
            );

            ffi::vk_queue_present_khr(
                &self.ctx,
                &self.queue,
                &mut self.swapchain,
                idx,
                &[self.render_finished_sem.handle],
            );
        }

        pub fn destroy(&mut self) {
            unsafe {
                let _ = self.ctx.device.device_wait_idle();
                self.ctx.device.destroy_pipeline(vk::Pipeline::from_raw(self.compute_pipeline.handle), None);
                self.ctx.device.destroy_pipeline_layout(vk::PipelineLayout::from_raw(self.pipeline_layout_handle), None);
                self.ctx.device.destroy_shader_module(
                    vk::ShaderModule::from_raw(self.shader_module.handle),
                    None,
                );
                self.ctx.device.destroy_descriptor_pool(vk::DescriptorPool::from_raw(self.descriptor_pool.handle), None);
                self.ctx.device.destroy_descriptor_set_layout(vk::DescriptorSetLayout::from_raw(self.descriptor_set_layout.handle), None);
                for iv in &self.image_views {
                    self.ctx
                        .device
                        .destroy_image_view(vk::ImageView::from_raw(iv.handle), None);
                }
                self.ctx.device.destroy_command_pool(
                    vk::CommandPool::from_raw(self.command_pool.handle),
                    None,
                );
                self.ctx
                    .device
                    .destroy_fence(vk::Fence::from_raw(self.in_flight_fence.handle), None);
                self.ctx.device.destroy_semaphore(
                    vk::Semaphore::from_raw(self.image_available_sem.handle),
                    None,
                );
                self.ctx.device.destroy_semaphore(
                    vk::Semaphore::from_raw(self.render_finished_sem.handle),
                    None,
                );
                self.ctx
                    .swapchain_loader
                    .destroy_swapchain(vk::SwapchainKHR::from_raw(self.swapchain.handle), None);
                self.ctx.surface_loader.destroy_surface(self.raw_surface, None);
                self.ctx.destroy();
            }
        }
    }

    fn spv_to_u32(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Application handler
// ═══════════════════════════════════════════════════════════════════════════

struct App {
    window: Option<std::sync::Arc<Window>>,
    state: Option<vulkan::VkState>,
}

impl App {
    fn capture_mouse(&mut self, capture: bool) {
        if let (Some(w), Some(vk)) = (&self.window, &mut self.state) {
            vk.mouse_captured = capture;
            if capture {
                let _ = w.set_cursor_grab(CursorGrabMode::Locked)
                    .or_else(|_| w.set_cursor_grab(CursorGrabMode::Confined));
                w.set_cursor_visible(false);
            } else {
                let _ = w.set_cursor_grab(CursorGrabMode::None);
                w.set_cursor_visible(true);
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = WindowAttributes::default().with_title("Verified Ray Tracer");
        let window = std::sync::Arc::new(
            event_loop
                .create_window(attrs)
                .expect("Failed to create window"),
        );
        self.window = Some(window.clone());
        self.state = Some(vulkan::VkState::new(window.clone()));
        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(ref mut vk) = self.state {
                    vk.resize(size.width, size.height);
                }
            }
            WindowEvent::MouseInput {
                state: winit::event::ElementState::Pressed,
                button: winit::event::MouseButton::Left,
                ..
            } => {
                self.capture_mouse(true);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    let pressed = event.state == winit::event::ElementState::Pressed;
                    if let Some(ref mut vk) = self.state {
                        if pressed {
                            vk.keys_held.insert(key);
                        } else {
                            vk.keys_held.remove(&key);
                        }

                        if pressed && !event.repeat {
                            match key {
                                KeyCode::Escape => {
                                    self.capture_mouse(false);
                                }
                                KeyCode::Equal | KeyCode::NumpadAdd => {
                                    vk.fov = (vk.fov - 0.05).max(0.1);
                                    eprintln!("FOV: {:.2}", vk.fov);
                                }
                                KeyCode::Minus | KeyCode::NumpadSubtract => {
                                    vk.fov = (vk.fov + 0.05).min(2.0);
                                    eprintln!("FOV: {:.2}", vk.fov);
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(ref mut vk) = self.state {
                    vk.render();
                }
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _device_id: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            if let Some(ref mut vk) = self.state {
                if vk.mouse_captured {
                    let sensitivity = 0.003f32;
                    vk.yaw += delta.0 as f32 * sensitivity;
                    vk.pitch -= delta.1 as f32 * sensitivity;
                    vk.pitch = vk.pitch.clamp(-1.5, 1.5);
                }
            }
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App {
        window: None,
        state: None,
    };
    let _ = event_loop.run_app(&mut app);
    if let Some(ref mut vk) = app.state {
        vk.destroy();
    }
}
