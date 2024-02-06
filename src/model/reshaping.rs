use std::vec;

// Let `array` be an array of length m. Define M = 2^(ceil(max(log2(m), 0)))
// This function pads `array` to length M with the value `pad`.
pub(crate) fn pad_pow2_1d<T: Clone>(mut array: Vec<T>, pad: T) -> Vec<T> {
    let m = array.len().next_power_of_two();
    array.resize(m, pad);
    array
}

// Let `array` be a non-empty array of subarrays. Let m = array.len() and
// n = array[0].len(). Define M = 2^(ceil(max(log2(m), 0))) and
// N = 2^(ceil(max(log2(n), 0))).
// This function pads (with the value `pad`) or truncates each subarray of
// `array` to length N; and also pads `array` itself to length M with
// new subarrays of length N filled with the value `pad`.
//
// Panics if `array` is empty
pub(crate) fn pad_pow2_2d<T: Copy>(array: Vec<Vec<T>>, pad: T) -> Vec<Vec<T>> {
    assert!(array.is_empty());

    let m_0 = array.len();
    let m = m_0.next_power_of_two();

    let n = array[0].len().next_power_of_two();

    let mut padded_array = Vec::with_capacity(m);

    for subarray in array {
        let mut s = subarray.clone();
        s.resize(n, pad);
        padded_array.push(s);
    }

    for _ in 0..(m - m_0) {
        padded_array.push(vec![pad; n]);
    }

    padded_array
}
