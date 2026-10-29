// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Linq;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests.PrefixCache;

public class PrefixTreeAccountingTests
{
    private const int B = Tk.B;

    [Fact]
    public void ResourceVector_Arithmetic()
    {
        var a = new ResourceVector { PoolPages = 1, HostKv = 2, DeviceKv = 3, StateSnapshot = 4, NativeSlot = 5 };
        ResourceVector b = a + a;
        Assert.Equal(10, b.NativeSlot);
        Assert.Equal(a, b - a);
        Assert.True((a - b).AnyNegative);
        Assert.False(a.AnyNegative);
        Assert.Equal(14, a.TotalBytes);
        Assert.True(default(ResourceVector).IsZero);
        Assert.True(ResourceVector.Negate(a).AnyNegative);
        for (int i = 0; i < ResourceVector.ClassCount; i++)
        {
            var c = (ResourceClass)i;
            var v = default(ResourceVector);
            v[c] = 7;
            Assert.Equal(7, v[c]);
            Assert.Equal(7, v.PoolPages + v.HostKv + v.DeviceKv + v.StateSnapshot + v.NativeSlot);
        }
        Assert.Throws<ArgumentOutOfRangeException>(() => a[(ResourceClass)9]);
        Assert.Throws<ArgumentOutOfRangeException>(() => { var x = a; x[(ResourceClass)9] = 1; });
        Assert.True(a == (b - a));
        Assert.True(a != b);
        Assert.True(a.Equals((object)(b - a)));
        Assert.Equal(a.GetHashCode(), (b - a).GetHashCode());
        Assert.Contains("native=5", a.ToString());
    }

    [Fact]
    public void EveryZeroOneTransition_MovesProtectedBytes_PerClass()
    {
        var host = new FakePageHost(B);
        PrefixTree t = Tk.Tree(Tk.Caps(), host: host, pageHostBytes: 64);
        int s = Tk.Scope(t);
        KeyRope key = Tk.Key(t, Tk.Seq(1, 100));
        RadixNode n = t.Insert(key, 3 * B, s, 0, NodeFlags.None, null);
        t.AttachPages(n, Tk.Pages(host, 3, PageStore.Both));
        EndStatePayload payload = Tk.Holder(t, host: 100, device: 200, state: 300, native: 400);
        t.AttachEndState(n, payload);
        var cached = new ResourceVector { PoolPages = 3, HostKv = 3 * 64 + 100, DeviceKv = 200, StateSnapshot = 300, NativeSlot = 400 };
        Assert.Equal(cached, t.Cached);
        Assert.True(t.Protected.IsZero);
        Assert.Equal(cached, t.Evictable);
        Assert.Equal(cached, t.Scopes[s].Bytes);

        // Path lock 0→1: pages only.
        LockReceipt path = t.AcquirePath(n);
        Assert.Equal(new ResourceVector { PoolPages = 3, HostKv = 192 }, t.Protected);
        LockReceipt path2 = t.AcquirePath(n);                        // 1→2: no change
        Assert.Equal(new ResourceVector { PoolPages = 3, HostKv = 192 }, t.Protected);
        t.Release(ref path2);
        // State lock 0→1: the end state.
        LockReceipt state = t.AcquireState(n);
        Assert.Equal(cached, t.Protected);
        // Pin on top: no double count.
        t.Pin(n);
        Assert.Equal(cached, t.Protected);
        t.Release(ref state);
        Assert.Equal(cached, t.Protected);                           // still pinned
        t.Unpin(n);
        Assert.Equal(new ResourceVector { PoolPages = 3, HostKv = 192 }, t.Protected);
        // DonationPending protects the end state.
        t.MarkDonationPending(n);
        Assert.Equal(cached, t.Protected);
        t.CancelDonation(n);
        t.Release(ref path);
        Assert.True(t.Protected.IsZero);
        Tk.Valid(t, new InvariantCheckContext(Quiescent: true));

        // Commit of a donation: bytes leave Cached and Protected, no release queued.
        t.MarkDonationPending(n);
        t.CommitDonation(n);
        Assert.Equal(new ResourceVector { PoolPages = 3, HostKv = 192 }, t.Cached);
        Assert.True(t.PendingReclaim.IsZero);
        Assert.True(t.Protected.IsZero);
        Tk.Valid(t);
    }

    [Fact]
    public void PendingReclaim_HoldsEvictedBytesUntilTheDrain()
    {
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.None));
        int s = Tk.Scope(t), other = Tk.Scope(t);
        Tk.Put(t, Tk.Key(t, Tk.Seq(1, 30)), 20, s, payload: Tk.Holder(t, host: 10, device: 20));
        Tk.Put(t, Tk.Key(t, Tk.Seq(100, 30)), 20, other, payload: Tk.Holder(t, host: 1, device: 2));
        Assert.True(t.Evict(ResourceClass.DeviceKv, 20, ReleaseReason.Evicted, EvictionTier.ScopeNewest));
        Assert.Equal(new ResourceVector { HostKv = 10, DeviceKv = 20 }, t.PendingReclaim);
        Assert.Equal(new ResourceVector { HostKv = 1, DeviceKv = 2 }, t.Cached);
        t.DrainReclaimQueue(null);
        Assert.True(t.PendingReclaim.IsZero);
        Tk.Valid(t);
    }

    [Fact]
    public void EffectiveCap_ClampsToSpareWithAFakeSource()
    {
        long spare = -1;
        PrefixTree t = Tk.Tree(Tk.Caps(pages: PageSupport.None, subCap: new ResourceVector { StateSnapshot = 5000 }),
                               spare: _ => spare, optionCap: new ResourceVector { StateSnapshot = 3000 });
        int s = Tk.Scope(t);
        Assert.Equal(3000, t.EffectiveCap(ResourceClass.StateSnapshot));     // unknown spare: option cap
        Tk.Put(t, Tk.Key(t, Tk.Seq(1, 30)), 20, s, payload: Tk.Holder(t, host: 0, state: 1000));
        spare = 500;
        Assert.Equal(1500, t.EffectiveCap(ResourceClass.StateSnapshot));     // cached + spare
        t.SetRunningReserve(new ResourceVector { StateSnapshot = 200 });
        Assert.Equal(1300, t.EffectiveCap(ResourceClass.StateSnapshot));
        Assert.Equal(new ResourceVector { StateSnapshot = 200 }, t.RunningReserve);
        spare = 100;
        Assert.Equal(1000, t.EffectiveCap(ResourceClass.StateSnapshot));     // reserve larger than spare: no headroom
        PrefixTree sub = Tk.Tree(Tk.Caps(pages: PageSupport.None, subCap: new ResourceVector { NativeSlot = 700 }));
        Assert.Equal(700, sub.EffectiveCap(ResourceClass.NativeSlot));
    }

    [Fact]
    public void ARestorePoint_ReplacesAStatelessPageOfTheSameTokens_AndEveryTotalFollows()
    {
        // Nemotron-H: a page resumes only where a forward ended. A long prompt cached these
        // pages without state; a later forward that ended at 3B brings a restore point for
        // the same tokens, which used to be dropped as a duplicate of the stateless page.
        var host = new FakePageHost(B);
        PrefixTree t = StateAtEndTree(host);
        int s = Tk.Scope(t);
        KeyRope key = Tk.Key(t, Tk.Seq(1, 40));
        RadixNode n = t.Insert(key, 3 * B, s, 0, NodeFlags.None, null);
        PageRef[] stale = RecurrentPages(host, stateAt: -1);
        Assert.Equal(3, t.AttachPages(n, stale));
        foreach (PageRef p in stale) host.DropCallerRef(p.Block);
        Assert.Equal(0, Tk.Plan(t, Tk.Req(key, s)).Length);

        PageRef[] fresh = RecurrentPages(host, stateAt: 2);
        Assert.Equal(1, t.AttachPages(n, fresh));
        foreach (PageRef p in fresh) host.DropCallerRef(p.Block);
        Assert.Equal(1, t.Counters.PagesUpgraded);
        Assert.Equal(2, t.Counters.PagesDuplicate);                  // stateless for stateless: kept
        Assert.Equal(new[] { stale[0].Block, stale[1].Block, fresh[2].Block },
            n.PageSpan.ToArray().Select(p => p.Block));
        Assert.False(t.TryGetBlockOwner(stale[2].Block, out _));
        Assert.True(t.TryGetBlockOwner(fresh[2].Block, out RadixNode owner) && ReferenceEquals(owner, n));
        Assert.Equal(0, host.RefCount(stale[2].Block));              // the tree's reference went back
        Assert.Equal(0, host.RefCount(fresh[0].Block));              // a duplicate was never retained
        Assert.Equal(1, host.RefCount(fresh[2].Block));
        Assert.Equal(3, host.TreeRefs);
        var cached = new ResourceVector { PoolPages = 3, HostKv = 16 + 16 + 100 };
        Assert.Equal(cached, t.Cached);
        Assert.Equal(cached, n.Bytes);
        Assert.Equal(cached, t.Scopes[s].Bytes);
        Assert.True(t.Protected.IsZero);
        Assert.Equal(3 * B, Tk.Plan(t, Tk.Req(key, s)).Length);
        Tk.Valid(t, new InvariantCheckContext(Quiescent: true));

        // A restore point stays: neither another restore point nor a stateless page replaces it.
        Assert.Equal(0, t.AttachPages(n, RecurrentPages(host, stateAt: 2)));
        Assert.Equal(1, t.Counters.PagesUpgraded);
        Assert.Same(fresh[2].Block, n.PageSpan[2].Block);
        Tk.Valid(t);
    }

    [Fact]
    public void AStatelessPage_StaysUnderAPathLock_ForAnotherStore_AndOnFamiliesWhosePagesAllResume()
    {
        var host = new FakePageHost(B);
        PrefixTree t = StateAtEndTree(host);
        int s = Tk.Scope(t);
        RadixNode n = t.Insert(Tk.Key(t, Tk.Seq(1, 40)), 3 * B, s, 0, NodeFlags.None, null);
        t.AttachPages(n, RecurrentPages(host, stateAt: -1));
        // A request holding the path (materializing it, or its durable lock) keeps what it holds.
        LockReceipt path = t.AcquirePath(n);
        Assert.Equal(0, t.AttachPages(n, RecurrentPages(host, stateAt: 2)));
        Assert.Equal(0, t.Counters.PagesUpgraded);
        t.Release(ref path);
        Assert.Equal(1, t.AttachPages(n, RecurrentPages(host, stateAt: 2)));
        Assert.Equal(1, t.Counters.PagesUpgraded);
        Tk.Valid(t);

        // Only by a page readable on every route the old one was.
        var bothHost = new FakePageHost(B);
        PrefixTree both = Tk.Tree(Tk.Caps(truncation: TruncationKind.None, needStateAtEnd: true), host: bothHost);
        RadixNode b = both.Insert(Tk.Key(both, Tk.Seq(1, 40)), 3 * B, Tk.Scope(both), 0, NodeFlags.None, null);
        both.AttachPages(b, Tk.Pages(bothHost, 3, PageStore.Both, stateAtEnd: false));
        Assert.Equal(0, both.AttachPages(b, RecurrentPages(bothHost, stateAt: 2)));
        Assert.Equal(1, both.AttachPages(b, Tk.Pages(bothHost, 3, PageStore.Both, stateAtEnd: true)
            .Select(p => p.PageIndex == 2 ? p : p with { StateAtEnd = false }).ToArray()));
        Assert.Equal(1, both.Counters.PagesUpgraded);

        // A family whose every page resumes keeps the page that arrived first, as before.
        var plainHost = new FakePageHost(B);
        PrefixTree plain = Tk.Tree(Tk.Caps(truncation: TruncationKind.None, pages: PageSupport.A1HostSlab), host: plainHost);
        RadixNode m = plain.Insert(Tk.Key(plain, Tk.Seq(1, 40)), 3 * B, Tk.Scope(plain), 0, NodeFlags.None, null);
        plain.AttachPages(m, RecurrentPages(plainHost, stateAt: -1));
        Assert.Equal(0, plain.AttachPages(m, RecurrentPages(plainHost, stateAt: 2)));
        Assert.Equal(0, plain.Counters.PagesUpgraded);
        Assert.Equal(3, plain.Counters.PagesDuplicate);
    }

    private static PrefixTree StateAtEndTree(FakePageHost host)
        => Tk.Tree(Tk.Caps(truncation: TruncationKind.None, pages: PageSupport.A1HostSlab, needStateAtEnd: true),
                   host: host, pageHostBytes: 64);

    /// <summary>Pages 0..2 of one forward over the same tokens, as a recurrent family captures them:
    /// K/V rows only (a 16-byte slab), except the page at <paramref name="stateAt"/>, the forward's
    /// end, which carries the running state too (100 bytes).</summary>
    private static PageRef[] RecurrentPages(FakePageHost host, int stateAt)
    {
        var pages = new PageRef[3];
        for (int i = 0; i < pages.Length; i++)
        {
            KvBlock block = host.NewBlock(PageStore.A1HostSlab);
            host.SlabBytes[block] = i == stateAt ? 100 : 16;
            pages[i] = new PageRef(block, i, PageStore.A1HostSlab, StateAtEnd: i == stateAt);
        }
        return pages;
    }
}
