pub(crate) type QSmallType = i8; // core type for quantised arithmetic: activation, matrix weights, etc.
pub(crate) type QLargeType = i32; // larger type for quantised arithmetic, used in FC and convolutional biases
pub(crate) type QScaleType = f32; // type for quantisation scales
                                  // TODO this one is likely exclusive to this module, reconsider visibility
pub(crate) type QScaleComputationType = f64; // larger precision type to compute the requantisation scale in some schemes
pub(crate) type QZeroPointType = QSmallType; // the quantisation zero-point has the same type as the output

pub(crate) struct QInfo {
    pub(super) scale: QScaleType,
    pub(crate) zero_point: QZeroPointType,
}

// TODO: this will probably change to inference-ready requantisation info
// Even what is being done now could be optimised by precomputing outside the
// evaluate function
pub(crate) struct FCQInfo {
    pub(crate) input_info: QInfo,
    pub(crate) weight_info: QInfo,
    // Bias requantisation information is not used (and is indeed directly
    // computable from the two above)
    pub(crate) output_info: QInfo,
}

pub(crate) enum RoundingScheme {
    NaiveNearestAwayFromZero,
}

pub(crate) fn requantise_fc(
    output: &[QLargeType],
    q_info: &FCQInfo,
    scheme: RoundingScheme,
) -> Vec<QSmallType> {
    match scheme {
        RoundingScheme::NaiveNearestAwayFromZero => requantise_fc_nnafz(output, q_info),
    }
}

fn requantise_fc_nnafz(output: &[QLargeType], q_info: &FCQInfo) -> Vec<QSmallType> {
    // 1. Computing scale
    // TODO In actual schemes, this will be decomposed as (int, shift)
    let (s_i, s_w, s_o) = (
        q_info.input_info.scale,
        q_info.weight_info.scale,
        q_info.output_info.scale,
    );
    let (s_i, s_w, s_o) = (
        s_i as QScaleComputationType,
        s_w as QScaleComputationType,
        s_o as QScaleComputationType,
    );
    let s = (s_i * s_w / s_o) as QScaleType;

    // 2. Requantise
    // TODO add rayon for parallelisation?
    output
        .iter()
        .map(|x| {
            let x = *x as QScaleType * s;
            let mut x = x.round() as QLargeType; // TODO which type to pick here? Should we check for overflows?
            x += q_info.output_info.zero_point as QLargeType;
            x.clamp(QSmallType::MIN as QLargeType, QSmallType::MAX as QLargeType) as QSmallType
        })
        .collect()
}
