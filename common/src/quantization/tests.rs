use super::*;
#[test]
fn test_nnafz_noop() {
    let output = vec![0, 1, 2, 3, 4, 5, 6, 7];
    let q_info = BMMQInfo {
        input_info: QInfo {
            scale: 1.0,
            zero_point: 0,
        },
        weight_info: QInfo {
            scale: 1.0,
            zero_point: 0,
        },
        output_info: QInfo {
            scale: 1.0,
            zero_point: 0,
        },
    };
    let expected = vec![0, 1, 2, 3, 4, 5, 6, 7];
    let actual = requantise_fc(&output, &q_info, RoundingScheme::NearestTiesAwayFromZero);
    assert_eq!(expected, actual);
}

#[test]
fn test_nnafz_halves() {
    // test when the output lands at .5 intervals
    let output = vec![-3, -2, -1, 0, 1, 2, 3];
    let q_info = BMMQInfo {
        input_info: QInfo {
            scale: 0.5,
            zero_point: 0,
        },
        weight_info: QInfo {
            scale: 1.0,
            zero_point: 0,
        },
        output_info: QInfo {
            scale: 1.0,
            zero_point: 0,
        },
    };
    let expected = vec![-2, -1, -1, 0, 1, 1, 2];
    let actual = requantise_fc(&output, &q_info, RoundingScheme::NearestTiesAwayFromZero);
    assert_eq!(expected, actual);
}

#[test]
fn test_frexp_positive_expon() {
    let num = 1.0;
    let expected = (0.5, 1);
    let actual = frexp(num);
    assert_eq!(expected, actual);
}

#[test]
fn test_frexp_negative_expon() {
    let num = 0.1;
    let expected = (0.8, -3);
    let actual = frexp(num);
    assert_eq!(expected, actual);
}

#[test]
fn test_frexp_zero_expon() {
    let num = 1.0 / 2.0 + 1.0 / ((1_i64 << 53) as f64);
    let expected = 0;
    let actual = frexp(num).1;
    assert_eq!(expected, actual);
}

#[test]
fn test_quantize_multiplier_zero() {
    let double_multiplier = 0_f64;
    let expected = (0, 0);
    let actual = quantize_multiplier(double_multiplier);
    assert_eq!(actual, expected);
}

#[test]
#[should_panic]
fn test_quantize_multiplier_non_negative_expon() {
    let double_multiplier = 1.0 / 2.0 + 1.0 / ((1_i64 << 53) as f64);
    let _ = quantize_multiplier(double_multiplier);
}

#[test]
fn test_quantize_multiplier_negative_expon() {
    let double_multiplier = 0.1;
    let expected = (1_717_986_918, 3);
    let actual = quantize_multiplier(double_multiplier);
    assert_eq!(expected, actual);
}

#[test]
#[should_panic]
fn test_ref_noop() {
    let (s_i, s_w, s_o) = (1.0, 1.0, 1.0);
    let double_mul = s_i * s_w / s_o;

    // panics because 1.0 = 0.5 * 2^1 and the exponent is positive.
    let _ = quantize_multiplier(double_mul);
}
