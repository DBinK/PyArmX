
from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar, cast

T = TypeVar("T")
U = TypeVar("U")


@dataclass(slots=True, frozen=True)
class Result(Generic[T]):
    """结果封装类, 模仿 Rust 的 Result 类"""
    
    value: T | None = None
    error: str | None = None
    code: str | None = None

    @classmethod
    def ok(cls, value: Optional[T] = None) -> "Result[T]":
        """创建成功结果。"""
        return cls(value=value)

    @classmethod
    def err(cls, error: str, code: Optional[str] = None) -> "Result[T]":
        """创建错误结果。"""
        return cls(error=error, code=code)

    @property
    def is_ok(self) -> bool:
        return self.error is None

    @property
    def is_err(self) -> bool:
        return self.error is not None

    def unwrap(self) -> T:
        """获取结果中的值。"""
        if self.is_err:
            raise RuntimeError(f"Unwrap failed: {self.error}")
        return cast(T, self.value)

    def unwrap_or(self, default: T) -> T:
        """获取结果中的值，如果结果为错误则返回默认值。"""
        if self.is_ok:
            return cast(T, self.value)
        return default

    def map(self, func: Callable[[T], U]) -> "Result[U]":
        """将结果中的值映射为另一个结果, 以实现链式调用。"""
        if self.is_ok:
            return Result.ok(func(cast(T, self.value)))
        return Result.err(cast(str, self.error), self.code)

    def __bool__(self) -> bool:
        """返回结果是否成功。"""
        return self.is_ok
    


if __name__ == "__main__":

    ret = Result.err("error", "A113")
    print(ret)

    ret = Result.err("error")
    print(ret)

    ret = Result.ok(1231)
    print(ret)