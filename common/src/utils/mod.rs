#[cfg(feature = "test-types")]
pub mod pcs_types;

#[cfg(feature = "test-types")]
pub mod test_sponge;

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
