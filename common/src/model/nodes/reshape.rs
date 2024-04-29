use ark_std::log2;

use crate::{model::tensor::SmallNIO, NIOTensor};

use super::{NodeOpsNative, NodeOpsPadded};

#[derive(Clone)]
pub struct ReshapeNode<ST: SmallNIO> {
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub padded_input_shape_log: Vec<usize>,
    pub padded_output_shape_log: Vec<usize>,
    phantom: std::marker::PhantomData<ST>,
}

impl<ST> NodeOpsNative<ST> for ReshapeNode<ST>
where
    ST: SmallNIO,
{
    fn shape(&self) -> (Vec<usize>, Vec<usize>) {
        (self.input_shape.clone(), self.output_shape.clone())
    }

    fn evaluate(&self, input: &NIOTensor<ST>) -> NIOTensor<ST> {
        let input = input.ref_small();

        // Sanity checks
        self.assert_valid_input(input.shape());

        let mut output = input.clone();
        output.reshape(self.output_shape.clone());
        NIOTensor::S(output)
    }

    fn type_name(&self) -> &'static str {
        "Reshape"
    }
}

impl<ST> NodeOpsPadded<ST> for ReshapeNode<ST>
where
    ST: SmallNIO,
{
    fn padded_shape_log(&self) -> Vec<usize> {
        self.padded_output_shape_log.clone()
    }

    fn padded_shape(&self) -> (Vec<usize>, Vec<usize>) {
        (
            self.padded_input_shape_log.iter().map(|x| 1 << x).collect(),
            self.padded_output_shape_log
                .iter()
                .map(|x| 1 << x)
                .collect(),
        )
    }

    // TODO I think this might be broken due to the failure of commutativity
    // between product and and nearest-geq-power-of-two
    fn padded_evaluate(&self, input: &NIOTensor<ST>) -> NIOTensor<ST> {
        let input = input.ref_small();

        // Sanity checks
        self.assert_valid_padded_input(input.shape());

        let mut unpadded_input = input.compact_resize(self.input_shape.clone(), ST::ZERO);

        // TODO only handles 2-to-1 reshapes, I think
        unpadded_input.reshape(self.output_shape.clone());

        NIOTensor::S(unpadded_input.compact_resize(self.padded_shape().1, ST::ZERO))
    }
}

impl<ST> ReshapeNode<ST>
where
    ST: SmallNIO,
{
    pub fn new(input_shape: Vec<usize>, output_shape: Vec<usize>) -> Self {
        assert_eq!(
            input_shape.iter().product::<usize>(),
            output_shape.iter().product::<usize>(),
            "Input and output shapes have a different number of entries",
        );

        // TODO does this break the invariant that the product of I and O coincides?
        let padded_input_shape_log = input_shape
            .iter()
            .map(|x| log2(x.next_power_of_two()) as usize)
            .collect();
        let padded_output_shape_log = output_shape
            .iter()
            .map(|x| log2(x.next_power_of_two()) as usize)
            .collect();

        Self {
            input_shape,
            output_shape,
            padded_input_shape_log,
            padded_output_shape_log,
            phantom: std::marker::PhantomData,
        }
    }
}

// TODO in constructor, add quantisation information checks? (s_bias = s_input * s_weight, z_bias = 0, z_weight = 0, etc.)
// TODO in constructor, check bias length matches appropriate matrix dimension
