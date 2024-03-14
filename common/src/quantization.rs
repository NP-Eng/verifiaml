use crate::model::qarray::InnerType;
use ark_std::fmt::Debug;

// TODO if we decide to make the model generic on the quantisation process
// types (which is probably correct now that the qtypes are generics), these
// will go away
// Type for quantisation scales
pub(crate) type QScaleType = f32;
// Larger precision type to compute the requantisation scale in some schemes
pub(crate) type QScaleComputationType = f64;

pub struct QInfo<ST> {
    pub scale: QScaleType,
    pub zero_point: ST,
}

// TODO: this will probably change to inference-ready requantisation info
// Even what is being done now could be optimised by precomputing outside the
// evaluate function
pub struct BMMQInfo<ST> {
    pub input_info: QInfo<ST>,
    pub weight_info: QInfo<ST>,
    // Bias requantisation information is not used (and is indeed directly
    // computable from the two above)
    pub output_info: QInfo<ST>,
}

pub enum RoundingScheme {
    NearestTiesAwayFromZero,
    NearestTiesEven,
}

pub fn requantise_fc<ST: InnerType, LT: InnerType>(
    output: &[LT],
    q_info: &BMMQInfo<ST>,
    scheme: RoundingScheme,
) -> Vec<ST>
where
    ST: InnerType + TryFrom<LT>,
    <ST as TryFrom<LT>>::Error: Debug,
    LT: InnerType + From<ST>,
{
    match scheme {
        RoundingScheme::NearestTiesAwayFromZero => requantise_fc_ntafz::<ST, LT>(output, q_info),
        RoundingScheme::NearestTiesEven => requantise_fc_nte::<ST, LT>(output, q_info),
    }
}

fn requantise_fc_ntafz<ST, LT>(output: &[LT], q_info: &BMMQInfo<ST>) -> Vec<ST>
where
    ST: InnerType + TryFrom<LT>,
    <ST as TryFrom<LT>>::Error: Debug,
    LT: InnerType + From<ST>,
{
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
            let x = LT::to_qscaletype(x) * s;
            let mut x = LT::from_qscaletype(x.round());
            x += LT::from(q_info.output_info.zero_point);
            ST::try_from(partial_ord_clamp(x, LT::from(ST::MIN), LT::from(ST::MAX))).unwrap()
        })
        .collect()
}

// The (unstable) method clamp comes from the trait Ord, which we cannot
// restrict InnerType to as we need f32 to implement the latter. Note that this
// method is not meaningfully defined for classes that genuinely do not
// implement Ord (total order relation) but only PartialOrd (partial order
// relation).
fn partial_ord_clamp<T: PartialOrd>(x: T, min: T, max: T) -> T {
    if x <= min {
        min
    } else if x >= max {
        max
    } else {
        x
    }
}

fn requantise_fc_nte<ST: InnerType, LT: InnerType>(output: &[LT], q_info: &BMMQInfo<ST>) -> Vec<ST>
where
    ST: InnerType + TryFrom<LT>,
    <ST as TryFrom<LT>>::Error: Debug,
    LT: InnerType + From<ST>,
{
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
            let x = LT::to_qscaletype(x) * s;
            let mut x = LT::from_qscaletype(x.round_ties_even()); // TODO which type to pick here? Should we check for overflows?
            x += LT::from(q_info.output_info.zero_point);
            ST::try_from(partial_ord_clamp(x, LT::from(ST::MIN), LT::from(ST::MAX))).unwrap()
        })
        .collect()
}

// This function is used to quantise model model inputs and its types are fixed
pub fn quantise_f32_u8_nne(values: &[f32], scale: QScaleType, zero: u8) -> Vec<u8> {
    values
        .iter()
        .map(|x| {
            ((((*x as QScaleType) / scale) + (zero as f32)).round_ties_even() as i32)
                .clamp(u8::MIN as i32, u8::MAX as i32) as u8
        })
        .collect()
}

#[cfg(test)]
mod tests {

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
}
