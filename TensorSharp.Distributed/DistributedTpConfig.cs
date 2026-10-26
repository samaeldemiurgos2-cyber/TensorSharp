using System;
using System.Linq;
using System.Net;
using System.Net.Sockets;

namespace TensorSharp.Distributed
{
    /// <summary>
    /// Peer topology shared by distributed tensor parallelism and layer splitting.
    /// Parsed from CLI arguments or environment variables.
    /// </summary>
    public sealed class DistributedTpConfig
    {
        /// <summary>This node's ID (0-based).</summary>
        public int NodeId { get; }

        /// <summary>Number of local GPUs on this node.</summary>
        public int LocalDegree { get; }

        /// <summary>TCP endpoints for every node, indexed by node ID.</summary>
        public IPEndPoint[] PeerEndpoints { get; }

        /// <summary>Total GPUs across all nodes.</summary>
        public int GlobalDegree => LocalDegree * PeerEndpoints.Length;

        public DistributedTpConfig(int nodeId, int localDegree, IPEndPoint[] peerEndpoints)
        {
            ArgumentNullException.ThrowIfNull(peerEndpoints);
            if (localDegree < 1)
                throw new ArgumentOutOfRangeException(nameof(localDegree));
            if (peerEndpoints.Length < 2)
                throw new ArgumentException("Distributed inference requires at least two peers.", nameof(peerEndpoints));
            if (peerEndpoints.Any(endpoint => endpoint == null || endpoint.Port < 1))
                throw new ArgumentException("Every peer must have a valid endpoint and nonzero port.", nameof(peerEndpoints));
            if (peerEndpoints.Distinct().Count() != peerEndpoints.Length)
                throw new ArgumentException("Peer endpoints must be distinct.", nameof(peerEndpoints));
            _ = checked(localDegree * peerEndpoints.Length);
            NodeId = nodeId;
            LocalDegree = localDegree;
            PeerEndpoints = peerEndpoints.Select(endpoint => new IPEndPoint(endpoint.Address, endpoint.Port)).ToArray();

            if ((uint)nodeId >= (uint)peerEndpoints.Length)
                throw new ArgumentOutOfRangeException(nameof(nodeId),
                    $"Node ID {nodeId} is out of range for {peerEndpoints.Length} nodes.");
        }

        /// <summary>
        /// Parse a comma-separated list of host:port pairs into IPEndpoints.
        /// Example: "192.168.1.10:9500,192.168.1.11:9500"
        /// </summary>
        public static IPEndPoint[] ParsePeers(string peersString)
        {
            if (string.IsNullOrWhiteSpace(peersString))
                throw new ArgumentException("Peers string is empty.", nameof(peersString));

            var parts = peersString.Split(',', StringSplitOptions.TrimEntries);
            var endpoints = new IPEndPoint[parts.Length];

            for (int i = 0; i < parts.Length; i++)
            {
                string part = parts[i];
                int separator = part.LastIndexOf(':');
                if (separator <= 0 || separator == part.Length - 1)
                    throw new FormatException(
                        $"Invalid peer endpoint '{part}'. Expected host:port or [IPv6]:port.");
                string host = part[..separator];
                if (host.StartsWith('[') && host.EndsWith(']'))
                    host = host[1..^1];
                else if (host.Contains(':'))
                    throw new FormatException($"IPv6 peer '{part}' must use [address]:port.");
                if (string.IsNullOrWhiteSpace(host))
                    throw new FormatException($"Empty host in peer endpoint '{part}'.");

                if (!int.TryParse(part[(separator + 1)..], out int port) || port < 1 || port > 65535)
                    throw new FormatException(
                        $"Invalid port in peer endpoint '{parts[i]}'.");

                if (IPAddress.TryParse(host, out var ip))
                {
                    endpoints[i] = new IPEndPoint(ip, port);
                }
                else
                {
                    // Resolve hostname.
                    var addresses = Dns.GetHostAddresses(host);
                    if (addresses.Length == 0)
                        throw new ArgumentException($"Could not resolve hostname '{host}'.");
                    // Prefer IPv4 consistently for hosts that publish both families.
                    endpoints[i] = new IPEndPoint(addresses.FirstOrDefault(address =>
                        address.AddressFamily == AddressFamily.InterNetwork) ?? addresses[0], port);
                }
            }

            return endpoints;
        }

        /// <summary>
        /// Try to build a <see cref="DistributedTpConfig"/> from environment variables.
        /// Returns null if the required variables are not set.
        ///
        /// Environment variables:
        ///   TENSORSHARP_TP_NODE_ID  — this node's ID (required for distributed mode)
        ///   TENSORSHARP_TP_PEERS    — comma-separated host:port list (required for distributed mode)
        /// </summary>
        public static DistributedTpConfig TryFromEnvironment(int localDegree)
        {
            string nodeIdStr = Environment.GetEnvironmentVariable("TENSORSHARP_TP_NODE_ID");
            string peersStr = Environment.GetEnvironmentVariable("TENSORSHARP_TP_PEERS");

            if (string.IsNullOrEmpty(nodeIdStr) || string.IsNullOrEmpty(peersStr))
                return null;

            if (!int.TryParse(nodeIdStr, out int nodeId))
                throw new FormatException($"TENSORSHARP_TP_NODE_ID must be an integer, got '{nodeIdStr}'.");

            var peers = ParsePeers(peersStr);
            return new DistributedTpConfig(nodeId, localDegree, peers);
        }
    }
}
