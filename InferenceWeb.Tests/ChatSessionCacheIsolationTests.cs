// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests;

[Collection(EngineEnvironmentCollection.Name)]
public class ChatSessionCacheIsolationTests
{
    [Fact]
    public void Default_RequiresLineageAndKeepsUnprovedRequestsSeparate()
    {
        using var env = new EnvScope();
        env.Set("TS_PREFIX_CACHE_ISOLATION", null);
        using var shared = new ChatSession("api", sharedAcrossConversations: true);
        Assert.Equal(IsolationMode.Lineage, ExecutionOptions.FromEnvironment().PrefixCacheIsolation);
        Assert.Equal("proven-scope", shared.ResolveCacheScope("proven-scope"));
        Assert.NotEqual(shared.ResolveCacheScope(null), shared.ResolveCacheScope(null));
    }

    [Fact]
    public void TrustedShared_IsExplicitAndDoesNotMergeNamedSessions()
    {
        using var env = new EnvScope();
        env.Set("TS_PREFIX_CACHE_ISOLATION", "shared");
        using var api = new ChatSession("api", sharedAcrossConversations: true);
        using var anotherApi = new ChatSession("other-api", sharedAcrossConversations: true);
        using var namedA = new ChatSession("a");
        using var namedB = new ChatSession("b");
        string scope = api.ResolveCacheScope(null);
        Assert.Equal(scope, api.ResolveCacheScope("previous-lineage"));
        Assert.Equal(scope, api.ResolveCacheScope(null));
        Assert.NotEqual(scope, anotherApi.ResolveCacheScope(null));
        Assert.NotEqual(scope, namedA.ResolveCacheScope(null));
        Assert.NotEqual(namedA.ResolveCacheScope(null), namedB.ResolveCacheScope(null));
        Assert.Equal(namedA.ResolveCacheScope(null), namedA.ResolveCacheScope("unrelated"));
        Assert.Contains("TS_PREFIX_CACHE_ISOLATION=shared", ExecutionOptions.FromEnvironment().DescribeOverrides());
    }

    [Fact]
    public void Strict_IgnoresAnInheritedScope()
    {
        using var env = new EnvScope();
        env.Set("TS_PREFIX_CACHE_ISOLATION", "strict");
        using var shared = new ChatSession("api", sharedAcrossConversations: true);
        Assert.NotEqual("old", shared.ResolveCacheScope("old"));
        Assert.NotEqual(shared.ResolveCacheScope("old"), shared.ResolveCacheScope("old"));
    }

    [Fact]
    public void UnknownIsolationMode_IsRejectedRatherThanSilentlySharing()
    {
        using var env = new EnvScope();
        env.Set("TS_PREFIX_CACHE_ISOLATION", "shraed");
        var error = Assert.Throws<ArgumentException>(ExecutionOptions.FromEnvironment);
        Assert.Contains("TS_PREFIX_CACHE_ISOLATION", error.Message);
    }
}
