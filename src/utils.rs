use ark_ff::Field;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};

pub(crate) fn fix_variables<F: Field>(
    poly: &DenseMultilinearExtension<F>,
    partial_point: &[F],
) -> DenseMultilinearExtension<F> {
    assert!(
        partial_point.len() <= poly.num_vars,
        "invalid size of partial point"
    );
    let nv = poly.num_vars;

    let mut poly = poly.evaluations.to_vec();
    let dim = partial_point.len();
    // evaluate single variable of partial point from right to left
    for i in 1..dim + 1 {
        let r = partial_point[i - 1];
        for b in 0..(1 << (nv - i)) {
            let left = poly[b << 1];
            let right = poly[(b << 1) + 1];
            poly[b] = left + r * (right - left);
        }
    }
    DenseMultilinearExtension::from_evaluations_slice(nv - dim, &poly[..(1 << (nv - dim))])
}
