using TensorSharp;
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

/// <summary>
/// Coverage for the Float32 path of the GGML Fill op. Contiguous views (including
/// a narrowed slice with a storage offset, the layout of Metal's in-place GDN
/// state) are filled with one span write; anything strided keeps the element
/// iterator. Each case checks the filled region and that nothing outside it moved.
/// </summary>
public class GgmlFillFloat32Tests
{
    private static GgmlAllocator NewCpuAllocator()
        => new GgmlAllocator(new GgmlContext(new[] { 0 }, GgmlBackendType.Cpu), 0);

    private static void Seed(Tensor t)
    {
        long n = t.ElementCount();
        var sizes = t.Sizes.ToArray();
        var idx = new long[sizes.Length];
        for (long i = 0; i < n; i++)
        {
            long rem = i;
            for (int d = sizes.Length - 1; d >= 0; d--) { idx[d] = rem % sizes[d]; rem /= sizes[d]; }
            t.SetElementAsFloat(1000f + i, idx);
        }
    }

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData(0f)]
    [InlineData(-2.5f)]
    public void Fill_ContiguousTensor_SetsEveryElement(float value)
    {
        var alloc = NewCpuAllocator();
        using var t = new Tensor(alloc, DType.Float32, 4, 3, 5);
        Seed(t);

        Ops.Fill(t, value);

        for (int a = 0; a < 4; a++)
            for (int b = 0; b < 3; b++)
                for (int c = 0; c < 5; c++)
                    Assert.Equal(value, t.GetElementAsFloat(a, b, c));
    }

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData(0f)]
    [InlineData(7f)]
    public void Fill_ContiguousNarrowWithStorageOffset_TouchesOnlyTheSlice(float value)
    {
        // Metal's in-place GDN layout: [attention output | state], with the state
        // a contiguous view at an offset into the shared backing storage.
        var alloc = NewCpuAllocator();
        using var backing = new Tensor(alloc, DType.Float32, 40);
        Seed(backing);
        using var slice = backing.Narrow(0, 12, 24);
        using var state = slice.View(2, 3, 4);

        Ops.Fill(state, value);

        for (int i = 0; i < 40; i++)
        {
            float expected = i >= 12 && i < 36 ? value : 1000f + i;
            Assert.Equal(expected, backing.GetElementAsFloat(i));
        }
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void Fill_StridedView_KeepsElementwiseSemantics()
    {
        // Narrowing an inner dimension leaves gaps between rows: only the view's
        // elements may change.
        var alloc = NewCpuAllocator();
        using var t = new Tensor(alloc, DType.Float32, 6, 8);
        Seed(t);
        using var view = t.Narrow(1, 2, 3);

        Ops.Fill(view, -1f);

        for (int r = 0; r < 6; r++)
            for (int c = 0; c < 8; c++)
            {
                float expected = c >= 2 && c < 5 ? -1f : 1000f + r * 8 + c;
                Assert.Equal(expected, t.GetElementAsFloat(r, c));
            }
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void Fill_TransposedView_FillsEveryElement()
    {
        var alloc = NewCpuAllocator();
        using var t = new Tensor(alloc, DType.Float32, 5, 7);
        Seed(t);
        using var transposed = t.Transpose();

        Ops.Fill(transposed, 3f);

        for (int r = 0; r < 5; r++)
            for (int c = 0; c < 7; c++)
                Assert.Equal(3f, t.GetElementAsFloat(r, c));
    }
}
