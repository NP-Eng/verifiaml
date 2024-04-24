pub mod nodes;
pub mod tensor;

use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_poly_commit::{LabeledCommitment, LabeledPolynomial, PolynomialCommitment};

use self::nodes::NodeOpsNative;
use self::tensor::{NIOTensor, SmallNIO};
use self::{nodes::NodeProof, tensor::Tensor};

pub type Poly<F> = DenseMultilinearExtension<F>;
pub type LabeledPoly<F> = LabeledPolynomial<F, DenseMultilinearExtension<F>>;

pub struct InferenceProof<F, S, PCS, ST>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
    ST: SmallNIO,
{
    // Model input tensors in plain
    pub inputs: Vec<NIOTensor<ST>>,

    // Model output tensors in plain
    pub outputs: Vec<NIOTensor<ST>>,

    // Commitments to each of the node values
    pub node_value_commitments: Vec<LabeledCommitment<PCS::Commitment>>,

    // Proofs of evaluation of each of the model's nodes
    pub node_proofs: Vec<NodeProof<F, S, PCS>>,

    // Proofs of opening of each of the model's inputs
    pub input_opening_proofs: Vec<PCS::Proof>,

    // Proofs of opening of each of the model's outputs
    pub output_opening_proofs: Vec<PCS::Proof>,
}

// TODO change the functions that receive vectors to receive slices instead whenever it makes sense

// TODO: for now, we require all nodes to use the same PCS; this might change
// in the future
pub struct Model<ST: SmallNIO> {
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub nodes: Vec<Box<dyn NodeOpsNative<ST>>>,
}

impl<ST: SmallNIO> Model<ST> {
    pub fn new(input_shape: Vec<usize>, nodes: Vec<Box<dyn NodeOpsNative<ST>>>) -> Self {
        // An empty model would cause panics later down the line e.g. when
        // determining the number of variables needed to commit to it.
        assert!(!nodes.is_empty(), "A model cannot have no nodes",);

        Self {
            input_shape,
            output_shape: nodes.last().unwrap().shape(),
            nodes,
        }
    }

    pub fn input_shape(&self) -> &Vec<usize> {
        &self.input_shape
    }

    pub fn evaluate(&self, input: Tensor<ST>) -> Tensor<ST> {
        self.nodes
            .iter()
            .fold(NIOTensor::S(input), |output, node| node.evaluate(&output))
            .unwrap_small()
    }
}
