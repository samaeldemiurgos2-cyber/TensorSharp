using System;

namespace TensorSharp;

/// <summary>Terminally fail a distributed session without releasing local device
/// allocations. Model tensors must be disposed before their owning local group.</summary>
public interface IDistributedExecutionGroup
{
    void Abort(Exception failure);
}
