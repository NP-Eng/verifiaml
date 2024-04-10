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
fn test_frexp() {
    let num = 1.0;
    let expected = (0.5, 1);
    let actual = frexp(num);
    println!("actual = {:?}", actual);
    assert_eq!(expected, actual);
}

#[test]
fn test_ref_noop() {
    let output = vec![0, 1, 2, 3, 4, 5, 6, 7];
    let (s_i, s_w, s_o) = (1.0, 1.0, 1.0);
    let double_mul = s_i * s_w / s_o;
    let output_zero_point = 0;

    let (effective_mul, effective_shift) = quantize_multiplier(double_mul);

    let expected = vec![0, 1, 2, 3, 4, 5, 6, 7];
    let actual =
        requantise_ref::<i8, i32>(&output, effective_mul, effective_shift, output_zero_point);
    assert_eq!(expected, actual);
}
