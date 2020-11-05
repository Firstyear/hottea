use ocl::{ProQue, Event, flags, SpatialDims};

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

const OCL_SRC: &str = r#"
    __kernel void f64sum(
        __global double *input_buffer,
        __global double *res_buffer,
        unsigned long step)
    {
        // Anything not marked as __local or __global is implied __private. This means
        // that intermediate values are stored per-gpu core which is much much much faster,
        // and part of the reason why this is so effective as a performance improvement, especially
        // for huge datasets.
        //
        // Our input buffer is a giant array like:
        // [0.0, 1.0, 2.0, 3.0, 4.0 ... ]
        //
        // The way we batch up work is that each work group will take it's work group id,
        // Which could be 0, 1, 2 ... WG_SIZE.
        // We then multiply idx by step to get the min so say step is 2, thread 0 will do
        // 0, 1, thread 1 will do 2, 3 etc ....
        size_t idx = get_local_id(0) * step;
        double acc = input_buffer[idx];

        for (size_t i = 1; i < step; i++) {
            acc = acc + input_buffer[idx + i];
        }

        // Write the result out to the output buffer.
        res_buffer[get_local_id(0)] = acc;
        // Done!
    }

"#;

// How many GPU kernels should be run in parallel?
#[cfg(not(test))]
const WG_SIZE: usize = 64;
#[cfg(test)]
const WG_SIZE: usize = 8;

struct HotTea {}

impl HotTea {
    pub fn new() -> Self {
        HotTea {
        }
    }

    pub(crate) fn do_mean_ocl(d: &[f64]) -> f64 {
        // Based on the size of our wg, how many elements per kernel?
        let step = if d.len() % WG_SIZE == 0 {
            d.len() / WG_SIZE
        } else {
            (d.len() / WG_SIZE) + 1
        };
        if step == 1 {
            // Each work group would only have to submit/manage 1 item, so there is no point
            // in submitting to the GPU - just burn CPU time instead.
            return Self::do_mean_cpu(d)
        }

        // Okay, so each work set now we know must do at least 2 ops or more. Begin by creating
        // a program-work-queue on the GPU.
        let pro_que = ProQue::builder()
            .src(OCL_SRC)
            .dims(
                // This is the dimension or parallel computation definition for the GPU.
                SpatialDims::new(
                    // Allocate 1 global work block.
                    Some(1),
                    // Which contains WG_SIZE work sets, IE we will run WG_SIZE items in parallel.
                    Some(WG_SIZE),
                    None,
                )
                .expect("Failed to create dimensions")
            )
            .build()
            .expect("failed to setup queue");

        // Setup the input buffer which is step * WG_SIZE, and allocate 0's at the end.
        let input_buffer = pro_que.buffer_builder()
            .len(WG_SIZE * step)
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
            input_buffer.write(d)
                .enew(&mut write_event)
                .block(false)
                .enq()
                .expect("Failed to write data");
        }

        // Create an output buffer, which is sized to the dimensions of the worksets.
        let res_buffer = pro_que.create_buffer::<f64>()
            .expect("failed to create buffer");

        // Create the reduce kernel. This is the work unit that will
        // be run on the GPU. Note the name f64sum matches the OCL
        // C definition at the top of the source. This is also where
        // we supply our arguments to the kernel.
        let kernel1 = pro_que.kernel_builder("f64sum")
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
            res_buffer.read(&mut res)
                .ewait(&mut k1_event)
                .block(true)
                .enq()
                .expect("Failed to queue result read");
        };

        // Now we can do the final fold on CPU to yield the mean.
        let d_len = f64::from(d.len() as u32);
        res.iter().fold(0.0, |acc, x| x + acc) / d_len
    }

    pub(crate) fn do_sd_ocl(d: &[f64], x: f64, c: f64) -> f64 {
        // TODO: Actually write the OCL version.
        Self::do_sd_cpu(d, x, c)
    }

    pub(crate) fn do_mean_cpu(d: &[f64]) -> f64 {
        let d_len = f64::from(d.len() as u32);
        d.iter().fold(0.0, |acc, x| x + acc) / d_len
    }

    pub(crate) fn do_sd_cpu(d: &[f64], x: f64, c: f64) -> f64 {
        let varience: f64 = d.iter().fold(0.0, |acc, i| {
            let diff = x - i;
            acc + (diff * diff)
        }) / (c - 1.0);

        varience.sqrt()
    }

    pub fn test(
        &self,
        x1: &[f64],
        x2: &[f64],
    ) -> f64 {
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

        let x1_mean = Self::do_mean_ocl(x1);
        let x2_mean = Self::do_mean_ocl(x2);

        // Calc the SD of a and b ... you know, just work it out :|
        let sd1 = Self::do_sd_ocl(x1, x1_mean, n1);
        let sd2 = Self::do_sd_ocl(x2, x2_mean, n2);

        let df: f64 = n1 + n2 - 2.0;

        // Start doing calculamations, beep boop.
        let poolvar = (
            ((n1 - 1.0) * (sd1 * sd1)) +
            ((n2 - 1.0) * (sd2 * sd2))
        ) / df;

        let ta = poolvar * (
                (1.0 / n1) + (1.0 / n2)
            );
        let t = (x1_mean - x2_mean) / ta.sqrt();

        println!("{:?}", t);
        t
    }
}

#[cfg(test)]
mod tests {
    use crate::HotTea;

    #[test]
    fn hottea_t_test_basic() {
        let t = HotTea::new();
        let a = [
                3.45,
                4.97,
                4.46,
                5.03,
                4.49,
                4.35,
                3.54,
                5.53,
                4.67,
                3.99,
                4.09,
                3.54,
                4.23,
                2.15,
                3.92,
                3.15,
                6.79,
                4.27,
                2.99,
                4.92,
                5.24,
                3.98,
                3.74,
                3.15,
                3.30,
                3.58,
                5.29,
                2.95,
                2.51,
                3.96,
            ];
        let b = [
                6.71,
                6.25,
                6.16,
                5.55,
                5.22,
                4.66,
                6.07,
                6.04,
                5.48,
                5.38,
                5.66,
                5.39,
                6.16,
                4.85,
                5.79,
                6.10,
                6.19,
                5.63,
                7.03,
                6.98,
                6.44,
                6.66,
                4.84,
                7.05,
                6.57,
                5.46,
                6.53,
                6.08,
                6.36,
                4.04,
            ];

        // Check some basic properties to be sure we are correct.
        // We know that the ocl and cpu impls are matching from below,
        // so we only use ocl here.
        let m_a = HotTea::do_mean_ocl(&a);
        let m_b = HotTea::do_mean_ocl(&b);
        let s_a = HotTea::do_sd_ocl(&a, m_a, 30.0);
        let s_b = HotTea::do_sd_ocl(&b, m_b, 30.0);

        // 0.734820
        println!("{:?}, {:?}", s_a, s_b);
        assert!(s_a == 0.9798757251721308);
        assert!(s_b == 0.7348204096803238);

        assert!(t.test(&a, &b) == -8.213501426846603);
    }

    #[test]
    fn hottea_mean_basic() {
        let vec1: Vec<f64> = vec![1.0f64; 128];
        let m_a = HotTea::do_mean_cpu(vec1.as_slice());
        let m_b = HotTea::do_mean_ocl(vec1.as_slice());
        assert!(m_a == 1.0);
        assert!(m_b == 1.0);

        let vec2: Vec<f64> = (0u32..128).map(|v| f64::from(v)).collect();
        let m_a = HotTea::do_mean_cpu(vec2.as_slice());
        let m_b = HotTea::do_mean_ocl(vec2.as_slice());
        assert!(m_a == 63.5);
        assert!(m_b == 63.5);
    }

    #[test]
    fn hottea_sd_basic() {
        let vec1: Vec<f64> = vec![1.0f64; 128];
        let l: u32 = 128;
        let m_a = HotTea::do_sd_cpu(vec1.as_slice(), 1.0, f64::from(l));
        let m_b = HotTea::do_sd_ocl(vec1.as_slice(), 1.0, f64::from(l));
        assert!(m_a == 0.0);
        assert!(m_b == 0.0);

        let vec2: Vec<f64> = vec![1.0, 8.0, -4.0, 9.0, 6.0];
        let l: u32 = 5;
        let m_a = HotTea::do_sd_cpu(vec2.as_slice(), 4.0, f64::from(l));
        assert!(m_a == 5.431390245600108);

        let vec3: Vec<f64> = (0u32..128).map(|v| f64::from(v)).collect();
        let l: u32 = 128;
        let m_a = HotTea::do_sd_cpu(vec3.as_slice(), 63.5, f64::from(l));
        let m_b = HotTea::do_sd_ocl(vec3.as_slice(), 63.5, f64::from(l));
        assert!(m_a == m_b);
        assert!(m_a == 37.094473981982816);
        assert!(m_b == 37.094473981982816);
    }
}




