use crate::config::DynError;

pub(crate) fn ensure(condition: bool, message: &str) -> Result<(), DynError> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}

pub(crate) fn ensure_eq<L, R>(left: &L, right: &R, message: &str) -> Result<(), DynError>
where
    L: PartialEq<R>,
{
    ensure(left == right, message)
}

pub(crate) fn ensure_ne<L, R>(left: &L, right: &R, message: &str) -> Result<(), DynError>
where
    L: PartialEq<R>,
{
    ensure(left != right, message)
}
