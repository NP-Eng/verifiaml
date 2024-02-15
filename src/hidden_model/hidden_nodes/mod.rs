use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;
use ark_std::rand::RngCore;

use crate::{
    error::VerificationError, model::nodes::NodeProof, proofs::InferenceProof, qarray::QArray,
    quantization::QSmallType, Poly,
};

use self::{
    hidden_fc::HiddenFCNode, hidden_loose_fc::HiddenLooseFCNode, hidden_relu::HiddenReLUNode,
    hidden_reshape::HiddenReshapeNode,
};

pub(crate) mod hidden_fc;
pub(crate) mod hidden_loose_fc;
pub(crate) mod hidden_relu;
pub(crate) mod hidden_reshape;

// mod parser;

// TODO: batched methods (e.g. for multiple evaluations)
// TODO: issue: missing info about size of the next output? Or reduplicate it?
// TODO way to handle generics more elegantly? or perhaps polynomials can be made ML directly?

/// A node of the model including its transition function to the next node(s).
/// It stores information about the transition (such as a matrix and bias, if
/// applicable), but not about about the specific values of its nodes: these
/// are handled by the methods only.
pub(crate) trait HiddenNodeOps<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    /// Returns the element-wise base-two logarithm of the padded node's
    /// output shape, i.e. the list of numbers of variables of the associated
    /// MLE
    fn padded_shape_log(&self) -> Vec<usize>;

    /// Returns the element-wise padded node's output shape
    fn padded_shape(&self) -> Vec<usize> {
        self.padded_shape_log()
            .into_iter()
            .map(|x| 1 << x)
            .collect()
    }

    /// The log of the number of output units of the padded node
    fn padded_num_units_log(&self) -> usize {
        self.padded_shape_log().iter().sum()
    }

    /// The number of output units of the padded node
    fn padded_num_units(&self) -> usize {
        self.padded_shape().iter().product()
    }

    /// Verify a proof of execution
    fn verify(&self, vk: &PCS::VerifierKey, proof: NodeProof) -> Result<(), VerificationError>;
}

pub(crate) enum HiddenNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    HiddenFC(HiddenFCNode<F, S, PCS>),
    HiddenLooseFC(HiddenLooseFCNode<F, S, PCS>),
    HiddenReLU(HiddenReLUNode<F, S, PCS>),
    HiddenReshape(HiddenReshapeNode<F, S, PCS>),
}

// A lot of this overlaps with the NodeOps trait and could be handled more
// elegantly by simply implementing the trait
impl<F, S, PCS> HiddenNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    // Print the type of the node. This cannot be cleantly achieved by deriving
    // Debug
    fn type_name(&self) -> &'static str {
        match self {
            HiddenNode::HiddenFC(_) => "FC",
            HiddenNode::HiddenLooseFC(_) => "LooseFC",
            HiddenNode::HiddenReLU(_) => "ReLU",
            HiddenNode::HiddenReshape(_) => "Reshape",
        }
    }
}

// A lot of this overlaps with the NodeOps trait and could be handled more
// elegantly by simply implementing the trait
impl<F, S, PCS> HiddenNodeOps<F, S, PCS> for HiddenNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    fn padded_shape_log(&self) -> Vec<usize> {
        match self {
            HiddenNode::HiddenFC(fc) => fc.padded_shape_log(),
            HiddenNode::HiddenLooseFC(fc) => fc.padded_shape_log(),
            HiddenNode::HiddenReLU(r) => r.padded_shape_log(),
            HiddenNode::HiddenReshape(r) => r.padded_shape_log(),
        }
    }

    fn verify(&self, vk: &PCS::VerifierKey, proof: NodeProof) -> Result<(), VerificationError> {
        match self {
            HiddenNode::HiddenFC(fc) => fc.verify(vk, proof),
            HiddenNode::HiddenLooseFC(fc) => fc.verify(vk, proof),
            HiddenNode::HiddenReLU(r) => r.verify(vk, proof),
            HiddenNode::HiddenReshape(r) => r.verify(vk, proof),
        }
    }
}
