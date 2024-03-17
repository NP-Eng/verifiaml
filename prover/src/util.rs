macro_rules! node_operation {
    ($self:expr, $method:ident, $($arg:expr),*) => {
        match $self {
            Node::BMM(node) => node.$method($($arg),*),
            Node::RequantiseBMM(node) => node.$method($($arg),*),
            Node::ReLU(node) => node.$method($($arg),*),
            Node::Reshape(node) => NodeOpsProve::<_, _, _, I, _>::$method(node, $($arg),*),
        }
    };
}
