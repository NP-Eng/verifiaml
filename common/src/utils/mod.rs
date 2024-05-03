use crate::{
    model::nodes::{
        requantize_bmm_float::RequantizeBMMFloatNode, requantize_bmm_ref::RequantizeBMMRefNode,
        requantize_bmm_single::RequantizeBMMSingleNode,
    },
    BMMRequantizationStrategy, Node,
};

#[cfg(feature = "test-types")]
pub mod pcs_types;

#[cfg(feature = "test-types")]
pub mod test_sponge;

// Convenience function to create a requantization Node variant depending
// on a chosen strategy. Only implemented for ST = i8, since the
// constructor of the reference implmementation (and therefore of single-round
// too) is only defined in this case
pub(crate) fn req_bmm_from_strategy(
    req_strategy: BMMRequantizationStrategy,
    inter_dim: usize,
    s_i: f32,
    z_i: i8,
    s_w: f32,
    z_w: i8,
    s_o: f32,
    z_o: i8,
) -> Node<i8> {
    match req_strategy {
        BMMRequantizationStrategy::Floating => Node::RequantizeBMMFloat(
            RequantizeBMMFloatNode::new(inter_dim, s_i, z_i, s_w, z_w, s_o, z_o),
        ),
        BMMRequantizationStrategy::Reference => {
            Node::RequantizeBMMRef(RequantizeBMMRefNode::new(inter_dim, s_i, s_w, s_o, z_o))
        }
        BMMRequantizationStrategy::SingleRound => {
            Node::RequantizeBMMSingle(RequantizeBMMSingleNode::new(inter_dim, s_i, s_w, s_o, z_o))
        }
    }
}

macro_rules! node_op {
    ($self:expr, $method:ident, $trait:ident) => {
        match $self {
            Node::BMM(node) => node.$method(),
            Node::RequantizeBMMFloat(node) => node.$method(),
            Node::RequantizeBMMRef(node) => node.$method(),
            Node::RequantizeBMMSingle(node) => node.$method(),
            Node::ReLU(node) => node.$method(),
            Node::Reshape(node) => $trait::<ST, _>::$method(node),
        }
    };
}
