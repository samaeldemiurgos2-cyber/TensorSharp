// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using Foundation;
using Microsoft.Maui.Handlers;
using Microsoft.Maui.Platform;
using WebKit;

namespace TensorAgent.Maui.Services;

/// <summary>Keep MAUI's navigation/cookie handling and report a lost chat renderer.</summary>
internal sealed class ChatWebViewNavigationDelegate : MauiWebViewNavigationDelegate
{
    private readonly Action _terminated;

    public ChatWebViewNavigationDelegate(WebViewHandler handler, Action terminated) : base(handler)
        => _terminated = terminated;

    // MAUI 10.0.110's Apple delegate does not implement this optional callback.
    // Memory pressure or a WebKit crash can kill the renderer while the native
    // window and its in-process inference/downloads are still perfectly healthy.
    [Export("webViewWebContentProcessDidTerminate:")]
    public void ContentProcessDidTerminate(WKWebView webView) => _terminated();
}
