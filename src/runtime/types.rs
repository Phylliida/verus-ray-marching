use vstd::prelude::*;
use verus_rational::runtime_rational::RuntimeRational;
use verus_linalg::runtime::vec3::RuntimeVec3;
use verus_geometry::runtime::point3::RuntimePoint3;
use crate::runtime::RationalModel;
use crate::types::*;

verus! {

// ---------------------------------------------------------------------------
// RuntimeRay3
// ---------------------------------------------------------------------------

/// Runtime ray: origin (Point3) + direction (Vec3) with ghost spec model.
pub struct RuntimeRay3 {
    pub origin: RuntimePoint3,
    pub dir: RuntimeVec3,
    pub model: Ghost<Ray3<RationalModel>>,
}

impl View for RuntimeRay3 {
    type V = Ray3<RationalModel>;
    open spec fn view(&self) -> Ray3<RationalModel> {
        self.model@
    }
}

impl RuntimeRay3 {
    pub open spec fn wf_spec(&self) -> bool {
        &&& self.origin.wf_spec()
        &&& self.dir.wf_spec()
        &&& self.origin@ == self@.origin
        &&& self.dir@ == self@.dir
    }

    pub fn new(origin: RuntimePoint3, dir: RuntimeVec3) -> (out: Self)
        requires
            origin.wf_spec(),
            dir.wf_spec(),
        ensures
            out.wf_spec(),
            out@.origin == origin@,
            out@.dir == dir@,
    {
        let ghost model = Ray3 { origin: origin@, dir: dir@ };
        RuntimeRay3 { origin, dir, model: Ghost(model) }
    }
}

// ---------------------------------------------------------------------------
// RuntimeSphere
// ---------------------------------------------------------------------------

/// Runtime sphere: center + squared radius.
pub struct RuntimeSphere {
    pub center: RuntimePoint3,
    pub radius_sq: RuntimeRational,
    pub model: Ghost<Sphere<RationalModel>>,
}

impl View for RuntimeSphere {
    type V = Sphere<RationalModel>;
    open spec fn view(&self) -> Sphere<RationalModel> {
        self.model@
    }
}

impl RuntimeSphere {
    pub open spec fn wf_spec(&self) -> bool {
        &&& self.center.wf_spec()
        &&& self.radius_sq.wf_spec()
        &&& self.center@ == self@.center
        &&& self.radius_sq@ == self@.radius_sq
    }

    pub fn new(center: RuntimePoint3, radius_sq: RuntimeRational) -> (out: Self)
        requires
            center.wf_spec(),
            radius_sq.wf_spec(),
        ensures
            out.wf_spec(),
            out@.center == center@,
            out@.radius_sq == radius_sq@,
    {
        let ghost model = Sphere { center: center@, radius_sq: radius_sq@ };
        RuntimeSphere { center, radius_sq, model: Ghost(model) }
    }
}

// ---------------------------------------------------------------------------
// RuntimePlane
// ---------------------------------------------------------------------------

/// Runtime plane: point + normal.
pub struct RuntimePlane {
    pub point: RuntimePoint3,
    pub normal: RuntimeVec3,
    pub model: Ghost<Plane<RationalModel>>,
}

impl View for RuntimePlane {
    type V = Plane<RationalModel>;
    open spec fn view(&self) -> Plane<RationalModel> {
        self.model@
    }
}

impl RuntimePlane {
    pub open spec fn wf_spec(&self) -> bool {
        &&& self.point.wf_spec()
        &&& self.normal.wf_spec()
        &&& self.point@ == self@.point
        &&& self.normal@ == self@.normal
    }

    pub fn new(point: RuntimePoint3, normal: RuntimeVec3) -> (out: Self)
        requires
            point.wf_spec(),
            normal.wf_spec(),
        ensures
            out.wf_spec(),
            out@.point == point@,
            out@.normal == normal@,
    {
        let ghost model = Plane { point: point@, normal: normal@ };
        RuntimePlane { point, normal, model: Ghost(model) }
    }
}

// ---------------------------------------------------------------------------
// RuntimeBox3
// ---------------------------------------------------------------------------

/// Runtime AABB: min + max points.
pub struct RuntimeBox3 {
    pub min: RuntimePoint3,
    pub max: RuntimePoint3,
    pub model: Ghost<Box3<RationalModel>>,
}

impl View for RuntimeBox3 {
    type V = Box3<RationalModel>;
    open spec fn view(&self) -> Box3<RationalModel> {
        self.model@
    }
}

impl RuntimeBox3 {
    pub open spec fn wf_spec(&self) -> bool {
        &&& self.min.wf_spec()
        &&& self.max.wf_spec()
        &&& self.min@ == self@.min
        &&& self.max@ == self@.max
    }

    pub fn new(min: RuntimePoint3, max: RuntimePoint3) -> (out: Self)
        requires
            min.wf_spec(),
            max.wf_spec(),
        ensures
            out.wf_spec(),
            out@.min == min@,
            out@.max == max@,
    {
        let ghost model = Box3 { min: min@, max: max@ };
        RuntimeBox3 { min, max, model: Ghost(model) }
    }
}

// ---------------------------------------------------------------------------
// RuntimeCylinder
// ---------------------------------------------------------------------------

/// Runtime cylinder: base center, axis direction, squared radius, half-height.
pub struct RuntimeCylinder {
    pub base_center: RuntimePoint3,
    pub axis_dir: RuntimeVec3,
    pub radius_sq: RuntimeRational,
    pub half_height: RuntimeRational,
    pub model: Ghost<Cylinder<RationalModel>>,
}

impl View for RuntimeCylinder {
    type V = Cylinder<RationalModel>;
    open spec fn view(&self) -> Cylinder<RationalModel> {
        self.model@
    }
}

impl RuntimeCylinder {
    pub open spec fn wf_spec(&self) -> bool {
        &&& self.base_center.wf_spec()
        &&& self.axis_dir.wf_spec()
        &&& self.radius_sq.wf_spec()
        &&& self.half_height.wf_spec()
        &&& self.base_center@ == self@.base_center
        &&& self.axis_dir@ == self@.axis_dir
        &&& self.radius_sq@ == self@.radius_sq
        &&& self.half_height@ == self@.half_height
    }

    pub fn new(
        base_center: RuntimePoint3,
        axis_dir: RuntimeVec3,
        radius_sq: RuntimeRational,
        half_height: RuntimeRational,
    ) -> (out: Self)
        requires
            base_center.wf_spec(),
            axis_dir.wf_spec(),
            radius_sq.wf_spec(),
            half_height.wf_spec(),
        ensures
            out.wf_spec(),
            out@.base_center == base_center@,
            out@.axis_dir == axis_dir@,
            out@.radius_sq == radius_sq@,
            out@.half_height == half_height@,
    {
        let ghost model = Cylinder {
            base_center: base_center@,
            axis_dir: axis_dir@,
            radius_sq: radius_sq@,
            half_height: half_height@,
        };
        RuntimeCylinder { base_center, axis_dir, radius_sq, half_height, model: Ghost(model) }
    }
}

} // verus!
