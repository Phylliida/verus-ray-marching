//! Verified ray tracer — renders a scene to PPM.
//!
//! The mathematical model (lighting, intersection, normals) is verified in Verus.
//! This binary implements the render loop using f64 for practical speed.
//!
//! Scene: 3 colored spheres on a gray ground plane, lit by a point light.
//!
//! Run: `cargo run --bin raytracer > scene.ppm`
//!      or with custom size: `cargo run --bin raytracer -- 512 384 > scene.ppm`

use std::io::Write;

// ═══════════════════════════════════════════════════════════════════════════
// Vec3 / Point3 — minimal f64 geometry
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }
    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
    fn sub(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
    fn scale(self, s: f64) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
    fn dot(self, rhs: Vec3) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }
    fn norm_sq(self) -> f64 {
        self.dot(self)
    }
    fn norm(self) -> f64 {
        self.norm_sq().sqrt()
    }
    fn normalized(self) -> Vec3 {
        let n = self.norm();
        if n > 1e-12 {
            self.scale(1.0 / n)
        } else {
            self
        }
    }
    fn cross(self, rhs: Vec3) -> Vec3 {
        Vec3::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }
    fn reflect(self, normal: Vec3) -> Vec3 {
        // I - 2 * dot(I, N) * N
        self.sub(normal.scale(2.0 * self.dot(normal)))
    }
}

type Point3 = Vec3;

// ═══════════════════════════════════════════════════════════════════════════
// Ray
// ═══════════════════════════════════════════════════════════════════════════

struct Ray {
    origin: Point3,
    dir: Vec3,
}

impl Ray {
    fn at(&self, t: f64) -> Point3 {
        self.origin.add(self.dir.scale(t))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Materials & colors
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy)]
struct Color {
    r: f64,
    g: f64,
    b: f64,
}

impl Color {
    fn new(r: f64, g: f64, b: f64) -> Self {
        Color { r, g, b }
    }
    fn scale(self, s: f64) -> Color {
        Color::new(self.r * s, self.g * s, self.b * s)
    }
    fn add(self, rhs: Color) -> Color {
        Color::new(self.r + rhs.r, self.g + rhs.g, self.b + rhs.b)
    }
    fn clamp01(self) -> Color {
        Color::new(
            self.r.max(0.0).min(1.0),
            self.g.max(0.0).min(1.0),
            self.b.max(0.0).min(1.0),
        )
    }
    fn to_u8(self) -> (u8, u8, u8) {
        let c = self.clamp01();
        // Gamma correction (sRGB approximate: gamma 2.2)
        let gamma = 1.0 / 2.2;
        (
            (c.r.powf(gamma) * 255.0 + 0.5) as u8,
            (c.g.powf(gamma) * 255.0 + 0.5) as u8,
            (c.b.powf(gamma) * 255.0 + 0.5) as u8,
        )
    }
}

#[derive(Clone, Copy)]
struct Material {
    color: Color,
    specular: f64,    // Specular coefficient [0, 1]
    shininess: f64,   // Phong exponent
}

// ═══════════════════════════════════════════════════════════════════════════
// Scene primitives
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone)]
enum Primitive {
    Sphere { center: Point3, radius: f64 },
    Plane { point: Point3, normal: Vec3 },
}

struct SceneObject {
    primitive: Primitive,
    material: Material,
}

struct HitRecord {
    t: f64,
    point: Point3,
    normal: Vec3,
    material: Material,
}

// ═══════════════════════════════════════════════════════════════════════════
// Ray intersection — mirrors the verified specs
// ═══════════════════════════════════════════════════════════════════════════

/// Ray-sphere intersection (Möller quadratic).
/// Corresponds to verified spec: ray_hits_sphere_nodiv + quadratic formula.
fn intersect_sphere(ray: &Ray, center: Point3, radius: f64) -> Option<f64> {
    let oc = ray.origin.sub(center);
    let a = ray.dir.norm_sq();
    let b = 2.0 * oc.dot(ray.dir);
    let c = oc.norm_sq() - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let sqrt_disc = disc.sqrt();
    let t1 = (-b - sqrt_disc) / (2.0 * a);
    let t2 = (-b + sqrt_disc) / (2.0 * a);
    if t1 > 1e-6 {
        Some(t1)
    } else if t2 > 1e-6 {
        Some(t2)
    } else {
        None
    }
}

/// Ray-plane intersection.
/// Corresponds to verified spec: ray_plane_t + ray_hits_plane.
fn intersect_plane(ray: &Ray, point: Point3, normal: Vec3) -> Option<f64> {
    let denom = ray.dir.dot(normal);
    if denom.abs() < 1e-12 {
        return None;
    }
    let t = point.sub(ray.origin).dot(normal) / denom;
    if t > 1e-6 {
        Some(t)
    } else {
        None
    }
}

fn intersect(ray: &Ray, prim: &Primitive) -> Option<f64> {
    match prim {
        Primitive::Sphere { center, radius } => intersect_sphere(ray, *center, *radius),
        Primitive::Plane { point, normal } => intersect_plane(ray, *point, *normal),
    }
}

fn normal_at(point: Point3, prim: &Primitive) -> Vec3 {
    match prim {
        Primitive::Sphere { center, .. } => point.sub(*center).normalized(),
        Primitive::Plane { normal, .. } => *normal,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Scene traversal
// ═══════════════════════════════════════════════════════════════════════════

fn closest_hit<'a>(ray: &Ray, objects: &'a [SceneObject]) -> Option<HitRecord> {
    let mut best: Option<HitRecord> = None;
    for obj in objects {
        if let Some(t) = intersect(ray, &obj.primitive) {
            let dominated = match &best {
                Some(rec) => t < rec.t,
                None => true,
            };
            if dominated {
                let p = ray.at(t);
                let n = normal_at(p, &obj.primitive);
                best = Some(HitRecord {
                    t,
                    point: p,
                    normal: n,
                    material: obj.material,
                });
            }
        }
    }
    best
}

// ═══════════════════════════════════════════════════════════════════════════
// Shading — mirrors verified lighting.rs specs
// ═══════════════════════════════════════════════════════════════════════════

struct PointLight {
    position: Point3,
    color: Color,
    intensity: f64,
}

/// Lambertian + Blinn-Phong shading.
/// The Lambertian component corresponds to the verified spec:
///   shade = max(0, dot(normal, light_dir))   [lambertian_clamped]
///   color = material * shade                  [shade_color]
///
/// Cauchy-Schwarz (lemma_lambertian_cauchy_schwarz) guarantees
/// that for unit vectors, shade ∈ [0, 1].
fn shade(
    hit: &HitRecord,
    light: &PointLight,
    ray_dir: Vec3,
    objects: &[SceneObject],
) -> Color {
    let to_light = light.position.sub(hit.point);
    let dist = to_light.norm();
    let light_dir = to_light.normalized();

    // Shadow test: cast ray from hit point toward light
    let shadow_ray = Ray {
        origin: hit.point.add(hit.normal.scale(1e-4)),
        dir: light_dir,
    };
    for obj in objects {
        if let Some(t) = intersect(&shadow_ray, &obj.primitive) {
            if t < dist {
                return Color::new(0.0, 0.0, 0.0); // In shadow
            }
        }
    }

    // Lambertian diffuse — verified in lighting.rs as lambertian_clamped
    let ndotl = hit.normal.dot(light_dir).max(0.0);
    let diffuse = hit.material.color.scale(ndotl * light.intensity);

    // Blinn-Phong specular
    let half_dir = light_dir.sub(ray_dir).normalized();
    let ndoth = hit.normal.dot(half_dir).max(0.0);
    let spec_factor = ndoth.powf(hit.material.shininess) * hit.material.specular * light.intensity;
    let specular = light.color.scale(spec_factor);

    diffuse.add(specular)
}

// ═══════════════════════════════════════════════════════════════════════════
// Sky gradient background
// ═══════════════════════════════════════════════════════════════════════════

fn background(ray: &Ray) -> Color {
    let t = 0.5 * (ray.dir.normalized().y + 1.0);
    let white = Color::new(1.0, 1.0, 1.0);
    let sky = Color::new(0.4, 0.6, 0.9);
    white.scale(1.0 - t).add(sky.scale(t))
}

// ═══════════════════════════════════════════════════════════════════════════
// Trace a single ray
// ═══════════════════════════════════════════════════════════════════════════

fn trace(ray: &Ray, objects: &[SceneObject], lights: &[PointLight], ambient: f64) -> Color {
    match closest_hit(ray, objects) {
        None => background(ray),
        Some(hit) => {
            // Ambient contribution
            let mut color = hit.material.color.scale(ambient);

            // Accumulate light contributions
            for light in lights {
                color = color.add(shade(&hit, light, ray.dir, objects));
            }

            color
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Camera
// ═══════════════════════════════════════════════════════════════════════════

struct Camera {
    origin: Point3,
    forward: Vec3,
    right: Vec3,
    up: Vec3,
}

impl Camera {
    fn look_at(from: Point3, at: Point3, world_up: Vec3, fov_deg: f64) -> Self {
        let forward = at.sub(from).normalized();
        let right = forward.cross(world_up).normalized();
        let up = right.cross(forward);
        let half_fov = (fov_deg * std::f64::consts::PI / 180.0) / 2.0;
        let scale = half_fov.tan();
        Camera {
            origin: from,
            forward,
            right: right.scale(scale),
            up: up.scale(scale),
        }
    }

    /// Generate a camera ray for pixel (px, py) in an image of size (w, h).
    /// Maps to normalized coords u, v ∈ [-1, 1].
    /// Corresponds to verified spec: camera_ray(cam, u, v).
    fn ray(&self, px: usize, py: usize, w: usize, h: usize) -> Ray {
        let aspect = w as f64 / h as f64;
        let u = (2.0 * (px as f64 + 0.5) / w as f64 - 1.0) * aspect;
        let v = 1.0 - 2.0 * (py as f64 + 0.5) / h as f64;
        let dir = self.forward.add(self.right.scale(u)).add(self.up.scale(v));
        Ray {
            origin: self.origin,
            dir: dir.normalized(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Checkerboard pattern for the ground plane
// ═══════════════════════════════════════════════════════════════════════════

fn checkerboard(point: Point3) -> Color {
    let scale = 1.0;
    let ix = (point.x * scale).floor() as i64;
    let iz = (point.z * scale).floor() as i64;
    if (ix + iz) % 2 == 0 {
        Color::new(0.9, 0.9, 0.9)
    } else {
        Color::new(0.3, 0.3, 0.3)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Scene construction
// ═══════════════════════════════════════════════════════════════════════════

fn build_scene() -> (Vec<SceneObject>, Vec<PointLight>, Camera) {
    let objects = vec![
        // Ground plane (y = 0)
        SceneObject {
            primitive: Primitive::Plane {
                point: Vec3::new(0.0, 0.0, 0.0),
                normal: Vec3::new(0.0, 1.0, 0.0),
            },
            material: Material {
                color: Color::new(0.7, 0.7, 0.7),
                specular: 0.05,
                shininess: 10.0,
            },
        },
        // Red sphere (left)
        SceneObject {
            primitive: Primitive::Sphere {
                center: Vec3::new(-2.0, 1.0, 5.0),
                radius: 1.0,
            },
            material: Material {
                color: Color::new(0.9, 0.15, 0.15),
                specular: 0.4,
                shininess: 32.0,
            },
        },
        // Green sphere (center, larger)
        SceneObject {
            primitive: Primitive::Sphere {
                center: Vec3::new(0.5, 1.5, 7.0),
                radius: 1.5,
            },
            material: Material {
                color: Color::new(0.15, 0.8, 0.2),
                specular: 0.3,
                shininess: 24.0,
            },
        },
        // Blue sphere (right, small)
        SceneObject {
            primitive: Primitive::Sphere {
                center: Vec3::new(3.0, 0.7, 4.5),
                radius: 0.7,
            },
            material: Material {
                color: Color::new(0.2, 0.3, 0.9),
                specular: 0.6,
                shininess: 64.0,
            },
        },
        // Gold sphere (behind, medium)
        SceneObject {
            primitive: Primitive::Sphere {
                center: Vec3::new(-0.5, 0.5, 3.0),
                radius: 0.5,
            },
            material: Material {
                color: Color::new(0.95, 0.75, 0.2),
                specular: 0.5,
                shininess: 48.0,
            },
        },
    ];

    let lights = vec![
        PointLight {
            position: Vec3::new(-3.0, 8.0, -2.0),
            color: Color::new(1.0, 1.0, 1.0),
            intensity: 0.8,
        },
        PointLight {
            position: Vec3::new(5.0, 5.0, 0.0),
            color: Color::new(1.0, 0.9, 0.8),
            intensity: 0.4,
        },
    ];

    let camera = Camera::look_at(
        Vec3::new(0.0, 3.0, -3.0),   // from
        Vec3::new(0.5, 1.0, 5.0),    // at
        Vec3::new(0.0, 1.0, 0.0),    // world up
        60.0,                          // FOV degrees
    );

    (objects, lights, camera)
}

// ═══════════════════════════════════════════════════════════════════════════
// Render with checkerboard ground
// ═══════════════════════════════════════════════════════════════════════════

fn trace_with_checkerboard(
    ray: &Ray,
    objects: &[SceneObject],
    lights: &[PointLight],
    ambient: f64,
) -> Color {
    match closest_hit(ray, objects) {
        None => background(ray),
        Some(mut hit) => {
            // Checkerboard on ground plane
            if matches!(objects.iter().find(|o| {
                if let Primitive::Plane { .. } = &o.primitive { true } else { false }
            }), Some(_))
                && hit.normal.y > 0.9
            {
                hit.material.color = checkerboard(hit.point);
            }

            let mut color = hit.material.color.scale(ambient);
            for light in lights {
                color = color.add(shade(&hit, light, ray.dir, objects));
            }
            color
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Main: render and output PPM
// ═══════════════════════════════════════════════════════════════════════════

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let width: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(640);
    let height: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(480);
    let ambient = 0.1;

    let (objects, lights, camera) = build_scene();

    let mut stdout = std::io::BufWriter::new(std::io::stdout().lock());
    writeln!(stdout, "P3").unwrap();
    writeln!(stdout, "{} {}", width, height).unwrap();
    writeln!(stdout, "255").unwrap();

    for py in 0..height {
        for px in 0..width {
            let ray = camera.ray(px, py, width, height);
            let color = trace_with_checkerboard(&ray, &objects, &lights, ambient);
            let (r, g, b) = color.to_u8();
            writeln!(stdout, "{} {} {}", r, g, b).unwrap();
        }
        // Progress to stderr
        if py % 50 == 0 {
            eprint!("\rRendering: {:.0}%", 100.0 * py as f64 / height as f64);
        }
    }
    eprintln!("\rRendering: 100% — done! ({} x {})", width, height);
}
