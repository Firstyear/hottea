use crate::HotTea;
use crate::bindings::*;
use std::convert::TryFrom;

#[no_mangle]
pub unsafe extern fn hottea_add(a: SEXP, b: SEXP) -> SEXP {
    let result: SEXP = Rf_protect(Rf_allocVector(REALSXP, 1));

    *(REAL(result)) = Rf_asReal(a) + Rf_asReal(b);

    Rf_unprotect_ptr(result);
    result
}

#[no_mangle]
pub unsafe extern fn hottea_mean(x: SEXP) -> SEXP {
    let result: SEXP = Rf_protect(Rf_allocVector(REALSXP, 1));

    let n: i32 = Rf_length(x);
    let n: usize = usize::try_from(n).expect("Failed to convert i32 -> usize");

    let data = std::slice::from_raw_parts(REAL(x), n);
    let ht = HotTea::new();
    *(REAL(result)) = ht.do_mean_cpu(data);

    Rf_unprotect_ptr(result);
    result
}





