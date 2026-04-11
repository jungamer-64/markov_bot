use crate::MarkovError;

pub(crate) fn ensure(condition: bool, message: &str) -> Result<(), MarkovError> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}

pub(crate) fn ensure_eq<L, R>(left: &L, right: &R, message: &str) -> Result<(), MarkovError>
where
    L: PartialEq<R>,
{
    ensure(left == right, message)
}
