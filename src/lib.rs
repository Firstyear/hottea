#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

// #![deny(warnings)]
#![warn(unused_extern_crates)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::unreachable)]
#![deny(clippy::await_holding_lock)]
#![deny(clippy::needless_pass_by_value)]
#![deny(clippy::trivially_copy_pass_by_ref)]

#[cfg(feature = "ocl_support")]
use ocl::{flags, Event, ProQue, SpatialDims};

#[macro_use]
extern crate log;

#[cfg(feature = "simd_support")]
use packed_simd::f64x8 as f64simd;

// Note: some rust types differ to C, but have equivalent memory layout which we
// require in this operation. They are:
//  RUST       C
//  u64        unsigned long
//  usize      size_t
//  f64        double
//  &[type]    <type> *
//  ()         void
//
// Additionally, OpenCL C has some specific elements that differ to normal C, but they
// are probably beyond the scope of this example.
// How many GPU kernels should be run in parallel?
#[cfg(not(test))]
const WG_SIZE: usize = 64;
#[cfg(test)]
const WG_SIZE: usize = 4;

// This module contains the OCL kernel sources
// we plan to use. See oclc.rs.
#[cfg(feature = "ocl_support")]
mod oclc;
// Our processed R headers
mod bindings;
// Our C interfaces.
pub mod external;

#[cfg(feature = "ocl_support")]
struct HotTea {
    pro_que: ProQue,
    wg_size: usize,
}

#[cfg(not(feature = "ocl_support"))]
struct HotTea {
}

impl HotTea {
    #[cfg(feature = "ocl_support")]
    pub fn new() -> Self {
        // Okay, so each work set now we know must do at least 2 ops or more. Begin by creating
        // a program-work-queue on the GPU.
        let pro_que = ProQue::builder()
            .src(oclc::SRC)
            .queue_properties(flags::CommandQueueProperties::OUT_OF_ORDER_EXEC_MODE_ENABLE)
            .build()
            .expect("failed to setup queue");

        let wg_size = WG_SIZE;

        info!("OCL device name -> {:?}", pro_que.device().name());
        info!("OCL device vendor -> {:?}", pro_que.device().vendor());
        info!("OCL device version -> {:?}", pro_que.device_version());

        HotTea { pro_que, wg_size }
    }

    #[cfg(not(feature = "ocl_support"))]
    pub fn new() -> Self {
        HotTea {}
    }

    #[cfg(feature = "ocl_support")]
    pub(crate) fn do_mean_ocl(&mut self, d: &[f64]) -> f64 {
        // Based on the size of our wg, how many elements per kernel?
        let step = if d.len() % self.wg_size == 0 {
            d.len() / self.wg_size
        } else {
            (d.len() / self.wg_size) + 1
        };
        if step == 1 {
            // Each work group would only have to submit/manage 1 item, so there is no point
            // in submitting to the GPU - just burn CPU time instead.
            return self.do_mean_cpu(d);
        }

        self.pro_que.set_dims(
            // This is the dimension or parallel computation definition for the GPU.
            SpatialDims::new(
                // Allocate 1 global work block.
                Some(1),
                // Which contains WG_SIZE work sets, IE we will run WG_SIZE items in parallel.
                Some(self.wg_size),
                None,
            )
            .expect("Failed to create dimensions"),
        );

        // Setup the input buffer which is step * WG_SIZE, and allocate 0's at the end.
        let input_buffer = self
            .pro_que
            .buffer_builder()
            .len(self.wg_size * step)
            // Define the input as read_only. This allows certain memory optimisations to occur
            // which increase performance ... for ... raisins >.>
            .flags(flags::MemFlags::new().read_only())
            // Fill with 0.0. This is because WG_SIZE * step may be greater than d.len(), so we
            // need to make sure we don't have junk memory at the tail of the buffer which would
            // cause problems.
            .fill_val(0.0)
            // Then now memcpy the content of d into the buffer.
            .build()
            .expect("failed to create buffer");

        let mut write_event = Event::empty();
        unsafe {
            input_buffer
                .write(d)
                .enew(&mut write_event)
                .block(false)
                .enq()
                .expect("Failed to write data");
        }

        // Create an output buffer, which is sized to the dimensions of the worksets.
        let res_buffer = self
            .pro_que
            .create_buffer::<f64>()
            .expect("failed to create buffer");

        // Create the reduce kernel. This is the work unit that will
        // be run on the GPU. Note the name f64sum matches the OCL
        // C definition at the top of the source. This is also where
        // we supply our arguments to the kernel.
        let kernel1 = self
            .pro_que
            .kernel_builder("f64sum")
            .arg(&input_buffer)
            .arg(&res_buffer)
            .arg(step as u64)
            .build()
            .expect("Fail to build kernel");

        // Queue the reduce to run. We provide k1_event because when
        // queued the GPU begins work immediately, but our host program
        // keeps going, so we need the k1_event item to allow the result
        // buffer read to wait on the sum to complete.
        let mut k1_event = Event::empty();
        unsafe {
            kernel1
                .cmd()
                .ewait(&mut write_event)
                .enew(&mut k1_event)
                .enq()
                .expect("failed to queue kernel");
        }

        // Allocate an output buffer we can read from
        let mut res: Vec<f64> = vec![0.0f64; res_buffer.len()];
        // Read the results. Note we ewait on the k1_event to ensure
        // f64sum is ordered before this operation , and we
        // block(true) to say we want this to actually force this
        // call to wait until the read is completed.
        unsafe {
            res_buffer
                .read(&mut res)
                .ewait(&mut k1_event)
                .block(true)
                .enq()
                .expect("Failed to queue result read");
        };

        // Now we can do the final fold on CPU to yield the mean.
        let d_len = f64::from(d.len() as u32);
        res.iter().fold(0.0, |acc, x| x + acc) / d_len
    }

    #[cfg(feature = "ocl_support")]
    pub(crate) fn do_sd_ocl(&mut self, d: &[f64], x: f64) -> f64 {
        let step = if d.len() % self.wg_size == 0 {
            d.len() / self.wg_size
        } else {
            (d.len() / self.wg_size) + 1
        };
        if step == 1 {
            return self.do_sd_cpu(d, x);
        }

        let c = f64::from(d.len() as u32);
        debug_assert!(c >= 2.0);

        /*
        self.pro_que.set_dims(
            SpatialDims::new(Some(1), Some(self.wg_size), None)
                .expect("Failed to create dimensions"),
        );
        */

        self.pro_que.set_dims(
            SpatialDims::new(Some(1), Some(self.wg_size), None)
                .expect("Failed to create dimensions"),
        );

        let input_buffer = self
            .pro_que
            .buffer_builder()
            .len(self.wg_size * step)
            .flags(flags::MemFlags::new().read_only())
            .fill_val(0.0)
            .build()
            .expect("failed to create buffer");

        let mut write_event = Event::empty();
        unsafe {
            input_buffer
                .write(d)
                .enew(&mut write_event)
                .block(false)
                .enq()
                .expect("Failed to write data");
        }

        // The intermediate result buffer.
        let res_buffer = self
            .pro_que
            .create_buffer::<f64>()
            .expect("failed to create buffer");

        let kernel1 = self
            .pro_que
            .kernel_builder("f64sd")
            .arg(&input_buffer)
            .arg(&res_buffer)
            .arg(x)
            .arg(step as u64)
            .arg(d.len() as u64)
            .build()
            .expect("Fail to build kernel");

        let mut k1_event = Event::empty();
        unsafe {
            kernel1
                .cmd()
                .ewait(&mut write_event)
                .enew(&mut k1_event)
                .enq()
                .expect("failed to queue kernel");
        }

        let mut res: Vec<f64> = vec![0.0f64; res_buffer.len()];
        unsafe {
            res_buffer
                .read(&mut res)
                .ewait(&mut k1_event)
                .block(true)
                .enq()
                .expect("Failed to queue result read");
        };

        let varience = res.iter().fold(0.0, |acc, i| i + acc) / (c - 1.0);
        let sd = varience.sqrt();
        debug!("sd result -> {:?}", sd);
        sd
    }

    #[cfg(not(feature = "simd_support"))]
    pub(crate) fn do_mean_cpu(&self, d: &[f64]) -> f64 {
        let d_len = f64::from(d.len() as u32);
        d.iter().fold(0.0, |acc, x| x + acc) / d_len
    }

    #[cfg(feature = "simd_support")]
    pub(crate) fn do_mean_cpu(&self, d: &[f64]) -> f64 {
        let d_len = f64::from(d.len() as u32);
        let (pre, main_set, rem) = unsafe { d.align_to::<f64simd>() };

        let main: f64 = main_set.iter().fold(0.0, |acc, i| acc + i.sum());

        let excess: f64 = pre.iter().chain(rem.iter()).fold(0.0, |acc, i| i + acc);

        (excess + main) / d_len
    }

    #[cfg(not(feature = "simd_support"))]
    pub(crate) fn do_sd_cpu(&self, d: &[f64], x: f64) -> f64 {
        let c = f64::from(d.len() as u32);
        let varience: f64 = d.iter().fold(0.0, |acc, i| {
            let diff = x - i;
            acc + (diff * diff)
        }) / (c - 1.0);

        varience.sqrt()
    }

    #[cfg(feature = "simd_support")]
    pub(crate) fn do_sd_cpu(&self, d: &[f64], x: f64) -> f64 {
        let c = f64::from(d.len() as u32);
        let (pre, main_set, rem) = unsafe { d.align_to::<f64simd>() };

        let x_simd = f64simd::splat(x);
        let main: f64 = main_set.iter().fold(0.0, |acc, i| {
            let diff = x_simd - *i;
            let inter = diff * diff;
            acc + inter.sum()
        });

        let excess: f64 = pre.iter().chain(rem.iter()).fold(0.0, |acc, i| {
            let diff = x - i;
            acc + (diff * diff)
        });

        let varience = (excess + main) / (c - 1.0);
        varience.sqrt()
    }

    #[inline(always)]
    #[cfg(feature = "ocl_support")]
    pub(crate) fn do_sd(&self, d: &[f64], x: f64) -> f64 {
        self.do_sd_ocl(d, x)
    }

    #[inline(always)]
    #[cfg(not(feature = "ocl_support"))]
    pub(crate) fn do_sd(&self, d: &[f64], x: f64) -> f64 {
        self.do_sd_cpu(d, x)
    }

    #[inline(always)]
    #[cfg(feature = "ocl_support")]
    pub(crate) fn do_mean(&self, d: &[f64]) -> f64 {
        self.do_mean_ocl(d)
    }

    #[inline(always)]
    #[cfg(not(feature = "ocl_support"))]
    pub(crate) fn do_mean(&self, d: &[f64]) -> f64 {
        self.do_mean_cpu(d)
    }

    pub fn t_test(&mut self, x1: &[f64], x2: &[f64]) -> f64 {
        // Stash len
        let n1 = x1.len();
        let n2 = x2.len();

        // Sanity check.
        assert!(n1 > 0);
        assert!(n2 > 0);

        // Convert everything to floats, this has to pass through u32 due to f64 only
        // allowing from u32. This *may* mean dataloss if a_len, b_len or df > UINT_MAX.
        let n1 = f64::from(n1 as u32);
        let n2 = f64::from(n2 as u32);

        let x1_mean = self.do_mean(x1);
        let x2_mean = self.do_mean(x2);

        // Calc the SD of a and b ... you know, just work it out :|
        let sd1 = self.do_sd(x1, x1_mean);
        let sd2 = self.do_sd(x2, x2_mean);

        let df: f64 = n1 + n2 - 2.0;

        // Start doing calculamations, beep boop.
        let poolvar = (((n1 - 1.0) * (sd1 * sd1)) + ((n2 - 1.0) * (sd2 * sd2))) / df;

        let ta = poolvar * ((1.0 / n1) + (1.0 / n2));
        let t = (x1_mean - x2_mean) / ta.sqrt();
        debug!("t_test result -> {:?}", t);
        t
    }
}

#[cfg(test)]
mod tests {
    use crate::HotTea;
    use std::time::Instant;

    #[test]
    fn hottea_t_test_basic() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut t = HotTea::new();
        let a = [
            3.45, 4.97, 4.46, 5.03, 4.49, 4.35, 3.54, 5.53, 4.67, 3.99, 4.09, 3.54, 4.23, 2.15,
            3.92, 3.15, 6.79, 4.27, 2.99, 4.92, 5.24, 3.98, 3.74, 3.15, 3.30, 3.58, 5.29, 2.95,
            2.51, 3.96,
        ];
        let b = [
            6.71, 6.25, 6.16, 5.55, 5.22, 4.66, 6.07, 6.04, 5.48, 5.38, 5.66, 5.39, 6.16, 4.85,
            5.79, 6.10, 6.19, 5.63, 7.03, 6.98, 6.44, 6.66, 4.84, 7.05, 6.57, 5.46, 6.53, 6.08,
            6.36, 4.04,
        ];

        // Check some basic properties to be sure we are correct.
        // We know that the ocl and cpu impls are matching from below,
        // so we only use ocl here.
        let m_a = t.do_mean_ocl(&a);
        let m_b = t.do_mean_ocl(&b);
        let s_a = t.do_sd_ocl(&a, m_a);
        let s_b = t.do_sd_ocl(&b, m_b);

        println!("{:?}, {:?}", s_a, s_b);
        // account for slight f64 differences between ocl && cpu.
        assert!(s_a == 0.9798757251721308 || s_a == 0.9798757251721310);
        assert!(s_b == 0.7348204096803238 || s_b == 0.7348204096803237);

        assert!(t.t_test(&a, &b) == -8.213501426846603);
    }

    #[test]
    fn hottea_mean_basic() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut t = HotTea::new();

        let vec1: Vec<f64> = vec![1.0f64; 128];
        let m_a = t.do_mean_cpu(vec1.as_slice());
        let m_b = t.do_mean_ocl(vec1.as_slice());
        assert!(m_a == 1.0);
        assert!(m_b == 1.0);

        let vec2: Vec<f64> = (0u32..128).map(|v| f64::from(v)).collect();
        let m_a = t.do_mean_cpu(vec2.as_slice());
        let m_b = t.do_mean_ocl(vec2.as_slice());
        assert!(m_a == 63.5);
        assert!(m_b == 63.5);
    }

    #[test]
    fn hottea_sd_basic() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut t = HotTea::new();

        let vec1: Vec<f64> = vec![1.0f64; 128];
        let t_a_1 = Instant::now();
        let m_a = t.do_sd_cpu(vec1.as_slice(), 1.0);
        let t_a_2 = Instant::now();
        info!("cpu t_a -> {:?}", t_a_2 - t_a_1);
        let t_b_1 = Instant::now();
        let m_b = t.do_sd_ocl(vec1.as_slice(), 1.0);
        let t_b_2 = Instant::now();
        info!("ocl t_b -> {:?}", t_b_2 - t_b_1);
        assert!(m_a == 0.0);
        assert!(m_b == 0.0);

        let vec2: Vec<f64> = vec![1.0, 8.0, -4.0, 9.0, 6.0];
        let m_a = t.do_sd_cpu(vec2.as_slice(), 4.0);
        assert!(m_a == 5.431390245600108);

        let vec3: Vec<f64> = (0u32..128).map(|v| f64::from(v)).collect();
        let t_a_1 = Instant::now();
        let m_a = t.do_sd_cpu(vec3.as_slice(), 63.5);
        let t_a_2 = Instant::now();
        info!("cpu t_a -> {:?}", t_a_2 - t_a_1);
        let t_b_1 = Instant::now();
        let m_b = t.do_sd_ocl(vec3.as_slice(), 63.5);
        let t_b_2 = Instant::now();
        info!("ocl t_b -> {:?}", t_b_2 - t_b_1);
        assert!(m_a == m_b);
        assert!(m_a == 37.094473981982816);
        assert!(m_b == 37.094473981982816);

        let data: [u32; 4] = [1024 << 3, 1024 << 7, 1024 << 14, 1 << 28];
        // let data: [u32; 1] = [1 << 28];
        for l in &data {
            let l = *l;
            println!("========= {}", l);
            let vec4: Vec<f64> = (0u32..l).map(|v| f64::from(v)).collect();
            let x: f64 = t.do_mean_cpu(vec4.as_slice());
            let mut acc = 0.0;
            let t_a_1 = Instant::now();
            for _i in 0..32 {
                let m_a = t.do_sd_cpu(vec4.as_slice(), x);
                acc += m_a;
            }
            let t_a_2 = Instant::now();
            debug!("{:?}", acc);
            info!("cpu t_a -> {:?}", t_a_2 - t_a_1);
            info!("cpu t_a -> {:?}", (t_a_2 - t_a_1).as_nanos());
            let t_b_1 = Instant::now();
            for _i in 0..32 {
                let m_b = t.do_sd_ocl(vec4.as_slice(), x);
                acc += m_b;
            }
            let t_b_2 = Instant::now();
            debug!("{:?}", acc);
            info!("ocl t_b -> {:?}", t_b_2 - t_b_1);
            info!("ocl t_b -> {:?}", (t_b_2 - t_b_1).as_nanos());
        }
    }
}
