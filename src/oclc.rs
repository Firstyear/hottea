pub const SRC: &str = r#"
// Begin OCL C.

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


// End OCL C
"#;
