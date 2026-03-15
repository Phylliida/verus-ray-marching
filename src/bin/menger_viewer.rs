//! Menger sponge fractal renderer using Vulkan compute.
//!
//! The GLSL compute shader mirrors the verified `ray_hits_fractal` algorithm
//! (iterative IFS descent with AABB pruning) from verus-ray-marching.
//!
//! Run: `cargo run --bin menger_viewer --features vulkan-backend`

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId, WindowAttributes},
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

    #[repr(C)]
    struct PushConstants {
        width: u32,
        height: u32,
        time: f32,
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
        // Compute-specific:
        compute_pipeline: vk::Pipeline,
        pipeline_layout: vk::PipelineLayout,
        descriptor_set_layout: vk::DescriptorSetLayout,
        descriptor_pool: vk::DescriptorPool,
        descriptor_sets: Vec<vk::DescriptorSet>,
        shader_module: RuntimeShaderModule,
        // Standard:
        command_pool: RuntimeCommandPool,
        command_buffers: Vec<RuntimeCommandBuffer>,
        image_available_sem: RuntimeSemaphore,
        render_finished_sem: RuntimeSemaphore,
        in_flight_fence: RuntimeFence,
        _format: vk::Format,
        width: u32,
        height: u32,
        start_time: Instant,
    }

    impl VkState {
        pub fn new(window: Arc<Window>) -> Self {
            let size = window.inner_size();
            let width = size.width.max(1);
            let height = size.height.max(1);

            // 1. Required surface extensions
            let display_handle = window.display_handle().unwrap();
            let surface_extensions =
                ash_window::enumerate_required_extensions(display_handle.as_raw())
                    .expect("Failed to get required surface extensions");

            let device_exts: Vec<*const i8> = vec![ash::khr::swapchain::NAME.as_ptr()];

            // 2. Create VulkanContext
            let ctx = unsafe {
                VulkanContext::new("menger_viewer", true, surface_extensions, &device_exts, 0)
            };

            // 3. Create surface
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

            // 4. Device + queue
            let mut dev = ffi::vk_create_device(&ctx, Ghost::assume_new());
            let queue = ffi::vk_get_device_queue(&ctx, 0, 0, Ghost::assume_new());

            // 5. Surface format — use UNORM (not SRGB) for storage image compatibility
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

            // 6. Swapchain — created with raw ash for STORAGE | COLOR_ATTACHMENT usage
            let image_count = 2u32;
            let swapchain_ci = vk::SwapchainCreateInfoKHR::default()
                .surface(raw_surface)
                .min_image_count(image_count)
                .image_format(format)
                .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
                .image_extent(vk::Extent2D { width, height })
                .image_array_layers(1)
                .image_usage(
                    vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                )
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(vk::PresentModeKHR::FIFO)
                .clipped(true);
            let raw_swapchain = unsafe {
                ctx.swapchain_loader.create_swapchain(&swapchain_ci, None)
            }
            .expect("Failed to create swapchain (STORAGE usage may not be supported)");

            // Wrap in RuntimeSwapchain for ffi compatibility
            let swapchain = RuntimeSwapchain {
                handle: raw_swapchain.as_raw(),
                state: Ghost::assume_new(),
            };

            // 7. Swapchain images + views
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

            // 8. Descriptor set layout: 1 storage image binding at COMPUTE stage
            let binding = vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE);
            let dsl_ci = vk::DescriptorSetLayoutCreateInfo::default()
                .bindings(std::slice::from_ref(&binding));
            let descriptor_set_layout = unsafe {
                ctx.device.create_descriptor_set_layout(&dsl_ci, None)
            }
            .expect("Failed to create descriptor set layout");

            // 9. Pipeline layout with push constants (12 bytes at COMPUTE stage)
            let dsl_handle = descriptor_set_layout.as_raw();
            let pipeline_layout = vk::PipelineLayout::from_raw(
                ffi::vk_create_pipeline_layout_push(
                    &ctx,
                    &[dsl_handle],
                    vk::ShaderStageFlags::COMPUTE.as_raw(),
                    0,
                    std::mem::size_of::<PushConstants>() as u32,
                ),
            );

            // 10. Shader module
            let spv_code = spv_to_u32(include_bytes!("shaders/menger.comp.spv"));
            let shader_module = ffi::vk_create_shader_module(&ctx, Ghost::assume_new(), &spv_code);

            // 11. Compute pipeline
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(vk::ShaderModule::from_raw(shader_module.handle))
                .name(c"main");
            let pipeline_ci = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(pipeline_layout);
            let compute_pipeline = unsafe {
                ctx.device
                    .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_ci], None)
            }
            .expect("Failed to create compute pipeline")[0];

            // 12. Descriptor pool + sets
            let pool_size = vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: image_count,
            };
            let pool_ci = vk::DescriptorPoolCreateInfo::default()
                .max_sets(image_count)
                .pool_sizes(std::slice::from_ref(&pool_size));
            let descriptor_pool = unsafe {
                ctx.device.create_descriptor_pool(&pool_ci, None)
            }
            .expect("Failed to create descriptor pool");

            // Allocate one descriptor set per swapchain image
            let layouts: Vec<vk::DescriptorSetLayout> =
                (0..image_count).map(|_| descriptor_set_layout).collect();
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&layouts);
            let descriptor_sets = unsafe {
                ctx.device.allocate_descriptor_sets(&alloc_info)
            }
            .expect("Failed to allocate descriptor sets");

            // Update each descriptor set to point at its swapchain image view
            for (i, &ds) in descriptor_sets.iter().enumerate() {
                let image_info = vk::DescriptorImageInfo {
                    sampler: vk::Sampler::null(),
                    image_view: vk::ImageView::from_raw(image_views[i].handle),
                    image_layout: vk::ImageLayout::GENERAL,
                };
                let write = vk::WriteDescriptorSet::default()
                    .dst_set(ds)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&image_info));
                unsafe { ctx.device.update_descriptor_sets(&[write], &[]) };
            }

            // 13. Command pool + buffers
            let command_pool = ffi::vk_create_command_pool(&ctx, Ghost::assume_new(), 0, true);
            let mut command_buffers = Vec::new();
            for _ in 0..image_count {
                let cb = ffi::vk_allocate_command_buffer(
                    &ctx,
                    Ghost::assume_new(),
                    command_pool.handle,
                );
                command_buffers.push(cb);
            }

            // 14. Sync objects
            let in_flight_fence = ffi::vk_create_fence(&ctx, Ghost::assume_new(), true);
            let image_available_sem = ffi::vk_create_semaphore(&ctx, Ghost::assume_new());
            let render_finished_sem = ffi::vk_create_semaphore(&ctx, Ghost::assume_new());

            eprintln!(
                "Menger viewer initialized: {}x{}, format {:?}",
                width, height, format
            );

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
                pipeline_layout,
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
                start_time: Instant::now(),
            }
        }

        pub fn resize(&mut self, width: u32, height: u32) {
            if width == 0 || height == 0 {
                return;
            }
            self.width = width;
            self.height = height;
            // TODO: recreate swapchain + descriptor sets for resize
        }

        pub fn render(&mut self) {
            // Wait for previous frame
            ffi::vk_wait_for_fences(
                &self.ctx,
                &mut self.in_flight_fence,
                Ghost::assume_new(),
                u64::MAX,
            );
            ffi::vk_reset_fences(&self.ctx, &mut self.in_flight_fence);

            // Acquire next image
            let idx = ffi::vk_acquire_next_image(
                &self.ctx,
                &mut self.swapchain,
                self.image_available_sem.handle,
                0,
                u64::MAX,
            );

            let cb = &mut self.command_buffers[idx as usize];
            let raw_cb = vk::CommandBuffer::from_raw(cb.handle);
            let image = vk::Image::from_raw(self.swapchain_images[idx as usize]);

            // Push constant data
            let elapsed = self.start_time.elapsed().as_secs_f32();
            let pc = PushConstants {
                width: self.width,
                height: self.height,
                time: elapsed,
            };
            let pc_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    &pc as *const PushConstants as *const u8,
                    std::mem::size_of::<PushConstants>(),
                )
            };

            // Record commands
            ffi::vk_begin_command_buffer(&self.ctx, cb);

            // Transition swapchain image: UNDEFINED → GENERAL (for compute write)
            let barrier_to_general = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            unsafe {
                self.ctx.device.cmd_pipeline_barrier(
                    raw_cb,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier_to_general],
                );
            }

            // Bind compute pipeline
            ffi::vk_cmd_bind_pipeline(
                &self.ctx,
                cb,
                vk::PipelineBindPoint::COMPUTE.as_raw() as u32,
                self.compute_pipeline.as_raw(),
            );

            // Bind descriptor set
            unsafe {
                self.ctx.device.cmd_bind_descriptor_sets(
                    raw_cb,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    &[self.descriptor_sets[idx as usize]],
                    &[],
                );
            }

            // Push constants
            ffi::ffi_cmd_push_constants(
                &self.ctx,
                cb.handle,
                self.pipeline_layout.as_raw(),
                vk::ShaderStageFlags::COMPUTE.as_raw(),
                0,
                pc_bytes,
            );

            // Dispatch compute — ceil(width/16) x ceil(height/16) x 1
            let group_x = (self.width + 15) / 16;
            let group_y = (self.height + 15) / 16;
            ffi::ffi_cmd_dispatch(&self.ctx, cb.handle, group_x, group_y, 1);

            // Transition swapchain image: GENERAL → PRESENT_SRC_KHR
            let barrier_to_present = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::empty())
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            unsafe {
                self.ctx.device.cmd_pipeline_barrier(
                    raw_cb,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier_to_present],
                );
            }

            ffi::vk_end_command_buffer(&self.ctx, cb);

            // Submit
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

            // Present
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
                self.ctx.device.destroy_pipeline(self.compute_pipeline, None);
                self.ctx
                    .device
                    .destroy_pipeline_layout(self.pipeline_layout, None);
                self.ctx.device.destroy_shader_module(
                    vk::ShaderModule::from_raw(self.shader_module.handle),
                    None,
                );
                self.ctx
                    .device
                    .destroy_descriptor_pool(self.descriptor_pool, None);
                self.ctx
                    .device
                    .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
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

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = WindowAttributes::default().with_title("Verified Menger Sponge");
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
