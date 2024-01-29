
pub(crate) type QSmallType = i8;                 // core type for quantised arithmetic: activation, matrix weights, etc.
pub(crate) type QLargeType = i32;                // larger type for quantised arithmetic, used in FC and convolutional biases
pub(crate) type Q_SCALE_TYPE = f32;                // type for quantisation scales
// TODO this one is likely exclusive to this module, reconsider visibility
pub(crate) type Q_SCALE_COMPUTATION_TYPE = f64;    // larger precision type to compute the requantisation scale in some schemes
pub(crate) type Q_ZERO_POINT_TYPE = QSmallType;  // the quantisation zero-point has the same type as the output

struct QInfo {
    scale: Q_SCALE_TYPE,
    zero_point: Q_ZERO_POINT_TYPE,
}

pub(crate) enum RoundingScheme {
    NaiveNearestAwayFromZero
}

pub(crate) fn requantise_fc(output: &[QLargeType], q_info: FCQInfo, scheme: RoundingScheme) -> Vec<QSmallType> {
    
    match scheme {
        RoundingScheme::NaiveNearestAwayFromZero => requantise_fc_nnafz(output, q_info),
    }

}

fn requantise_fc_nnafz(output: &[QSmallType], q_info: FCQInfo) -> Vec<QSmallType> {

    // 1. Computing scale
    // TODO In actual schemes, this will be decomposed as (int, shift)
    let (s_i, s_w, s_o) = (q_info.input_info.scale, q_info.weight_info.scale, q_info.output_info.scale);
    let (s_i, s_w, s_o) = (s_i as Q_SCALE_COMPUTATION_TYPE, s_w as Q_SCALE_COMPUTATION_TYPE, s_o as Q_SCALE_COMPUTATION_TYPE);
    let s = s_i * s_w / s_o as Q_SCALE_TYPE;

    // 2. Requantise
    // TODO add rayon for parallelisation?
    output.iter().map(|x| 
        {
            let x = x as Q_SCALE_TYPE * s;
            let mut x = x.round() as QLargeType; // TODO which type to pick here? Should we check for overflows?
            x += q_info.output_info.zero_point as QLargeType;
            x.clamp(QSmallType::MIN, QSmallType::MAX) as QSmallType
        }
    ).collect()

}
