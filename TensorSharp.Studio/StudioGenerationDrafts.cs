// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Text.Json.Serialization;

namespace TensorSharp.Studio;

/// <summary>A reviewable shot request authored while a conversational model is loaded.
/// Starting it requires a deliberate model selection; no model is automatically switched.</summary>
public sealed class StudioGenerationDraft
{
    public string Id { get; set; } = StudioJson.NewId();
    public string ProjectId { get; set; } = "";
    public long ExpectedRevision { get; set; }
    public string Summary { get; set; } = "";
    public StudioGenerationRequest Request { get; set; } = new();
    /// <summary>For a plan that continues a video: the frame of that video the shot starts on. The frame is taken (as a
    /// still, <see cref="StudioService.ExtractFrameAsync"/>) when the plan is started, and becomes the request's first frame,
    /// so planning never changes the library.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public StudioFrameSelector? FirstFrameFrom { get; set; }
    public string? JobId { get; set; }
    public DateTimeOffset CreatedAt { get; set; } = DateTimeOffset.UtcNow;
}

public sealed partial class StudioService
{
    string DraftDirectory(string projectId) => StudioPaths.SafeChild(ProjectDirectory(projectId), "generation-drafts");
    string DraftPath(string projectId, string draftId) => StudioPaths.SafeChild(DraftDirectory(projectId), StudioPaths.Id(draftId) + ".json");

    public Task<StudioGenerationDraft> CreateGenerationDraftAsync(string projectId, long expectedRevision,
        StudioGenerationRequest request, string summary, CancellationToken cancellationToken = default) =>
        CreateGenerationDraftAsync(projectId, expectedRevision, request, summary, null, cancellationToken);

    /// <param name="firstFrameFrom">The video frame a plan that continues a video starts on; the request then names no
    /// first frame of its own.</param>
    public async Task<StudioGenerationDraft> CreateGenerationDraftAsync(string projectId, long expectedRevision,
        StudioGenerationRequest request, string summary, StudioFrameSelector? firstFrameFrom, CancellationToken cancellationToken = default)
    {
        if (request == null || request.TemporalConditions == null || request.ReferenceAssetIds == null || request.ReferenceVideoAssetIds == null || request.ProjectId == null
            || request.Prompt == null || summary == null || request.TemporalConditions.Any(condition => condition == null) ||
            request.ReferenceAssetIds.Any(id => id == null) || request.ReferenceVideoAssetIds.Any(id => id == null))
            throw new StudioException("invalid_draft", "Generation plans require non-null request fields, conditioning arrays, and a summary.");
        request = StudioJson.Clone(request);
        await projectGate.WaitAsync(cancellationToken);
        try
        {
            var project = LoadDocument(projectId).Project; CheckRevision(project, expectedRevision);
            if (request.ProjectId.Length > 0 && request.ProjectId != projectId)
                throw new StudioException("project_conflict", "A generation plan cannot target another project.");
            request.ProjectId = projectId; StudioPaths.Id(request.RequestId);
            if (string.IsNullOrWhiteSpace(request.Prompt) || request.Prompt.Length > 16000 || summary.Length is < 1 or > 1000)
                throw new StudioException("invalid_draft", "Generation plans require a bounded prompt and short summary.");
            if (request.Width != 0 && request.Width is < 32 or > 2048 || request.Height != 0 && request.Height is < 32 or > 2048 || request.FrameCount is < 0 or > 257 || request.Steps is < 0 or > 100 || request.Fps is < 0 or > 240 || !float.IsFinite(request.CfgScale) || request.CfgScale is < 0 or > 30)
                throw new StudioException("invalid_draft", "Suggested generation settings exceed Studio admission limits.");
            if (firstFrameFrom != null)
            {
                if (request.ConditioningAssetId != null)
                    throw new StudioException("invalid_draft", "A plan that continues a video names that video in firstFrameFrom; leave conditioningAssetId empty.");
                var video = project.Assets.FirstOrDefault(a => a.Id == StudioPaths.Id(firstFrameFrom.AssetId))
                    ?? throw new StudioException("asset_not_found", "The video to continue does not belong to the project.");
                if (video.Kind != "video") throw new StudioException("invalid_frame_source", "firstFrameFrom names a video to continue; a photo is a conditioningAssetId.");
                if (firstFrameFrom.Frame < StudioFrameSource.LastFrame || firstFrameFrom.Frame > Options.MaxTimelineFrames)
                    throw new StudioException("invalid_frame", "firstFrameFrom.frame must be \"last\" or a frame from 0 within the supported timeline range.");
                // Checked again exactly when the frame is taken; the recorded length can include a longer soundtrack.
                if (firstFrameFrom.Frame != StudioFrameSource.LastFrame && firstFrameFrom.Frame >= Math.Ceiling(video.DurationSeconds * project.FrameRate.Value - 1e-9))
                    throw new StudioException("invalid_frame", $"Frame {firstFrameFrom.Frame} is past the end of this video. Choose an earlier frame, or its last frame.");
                firstFrameFrom = new() { AssetId = firstFrameFrom.AssetId, Frame = firstFrameFrom.Frame };
            }
            // The mode must match its inputs now, the frame still to be taken counting as the first frame; whether a
            // checkpoint runs it is checked when the plan starts.
            var planned = StudioJson.Clone(request);
            if (firstFrameFrom != null) planned.ConditioningAssetId = firstFrameFrom.AssetId;
            _ = StudioGenerationModes.Resolve(planned); request.Mode = planned.Mode;
            // Assets are capabilities scoped to this project, even before a model can validate their exact combination.
            _ = ResolveConditioning(request, project);
            Directory.CreateDirectory(DraftDirectory(projectId));
            if (Directory.EnumerateFiles(DraftDirectory(projectId), "*.json").Take(101).Count() >= 100)
                throw new StudioException("draft_limit", "This project already contains 100 generation plans.");
            var draft = new StudioGenerationDraft { ProjectId = projectId, ExpectedRevision = expectedRevision, Summary = summary, Request = request, FirstFrameFrom = firstFrameFrom };
            StudioStorage.Save(DraftPath(projectId, draft.Id), draft); return draft;
        }
        finally { projectGate.Release(); }
    }

    public async Task<IReadOnlyList<StudioGenerationDraft>> ListGenerationDraftsAsync(string projectId, CancellationToken cancellationToken = default)
    {
        await projectGate.WaitAsync(cancellationToken);
        try
        {
            _ = LoadDocument(projectId);
            if (!Directory.Exists(DraftDirectory(projectId))) return [];
            return Directory.EnumerateFiles(DraftDirectory(projectId), "*.json").Take(100)
                .Select(path => StudioStorage.Read<StudioGenerationDraft>(DraftPath(projectId, Path.GetFileNameWithoutExtension(path))))
                .OrderBy(d => d.CreatedAt).ToArray();
        }
        finally { projectGate.Release(); }
    }

    public async Task<StudioJob> StartGenerationDraftAsync(string projectId, string draftId, long expectedRevision,
        string modelId, CancellationToken cancellationToken = default)
    {
        StudioGenerationDraft draft;
        await projectGate.WaitAsync(cancellationToken);
        try
        {
            var project = LoadDocument(projectId).Project;
            draft = StudioStorage.Read<StudioGenerationDraft>(DraftPath(projectId, draftId));
            if (draft.ProjectId != projectId) throw new StudioException("project_conflict", "Generation plan belongs to another project.");
            if (string.IsNullOrWhiteSpace(modelId)) throw new StudioException("model_required", "Explicitly select and load a video model before starting this plan.");
            if (!string.IsNullOrEmpty(draft.Request.ModelId) && draft.Request.ModelId != modelId)
                throw new StudioException("model_conflict", "The plan names another model. Create a new plan to change it explicitly.");
            // A start that is retried returns the plan's job, even after the frame that start took moved the project on.
            if (draft.JobId != null)
                lock (jobSync)
                    if (jobs.TryGetValue(draft.JobId, out var started) && started.Job.ProjectId == projectId) return Present(started, DateTimeOffset.UtcNow);
            CheckRevision(project, expectedRevision);
            draft.Request.ModelId = modelId;
        }
        finally { projectGate.Release(); }
        // A plan that continues a video starts on that video's frame, taken now (a frame taken before is reused). The loaded
        // model and the plan's other inputs are checked first, with the video standing in for the frame, so a plan refused by
        // those checks leaves the library unchanged; the model's own resolvers run after the frame is taken, as for any take.
        if (draft.FirstFrameFrom is { } from && draft.Request.ConditioningAssetId == null)
        {
            if (generation == null) throw new StudioException("generation_unavailable", "No native generation adapter is connected to this host.");
            var planned = StudioJson.Clone(draft.Request); planned.ConditioningAssetId = from.AssetId;
            await AdmitAsync(planned, cancellationToken);
            // The plan's other photos still belong to the project, and a negative prompt is not sent where guidance 1 runs none.
            _ = ResolveConditioning(draft.Request, await GetProjectAsync(projectId, cancellationToken));
            if (planned.CfgScale == 1 && !string.IsNullOrWhiteSpace(planned.NegativePrompt))
                throw new StudioException("unsupported_negative_prompt", "Negative prompts do not run at CFG 1. Clear the negative prompt rather than silently ignoring it.");
            draft.Request.ConditioningAssetId = (await ExtractFrameAsync(projectId, from.AssetId, from.Frame, cancellationToken)).AssetId;
        }
        // Admission resolves model defaults and validates the exact checkpoint before queueing.
        // RequestId makes retries idempotent even if the host exits before saving JobId below.
        var job = await StartGenerationAsync(draft.Request, cancellationToken);
        await projectGate.WaitAsync(cancellationToken);
        try { draft.JobId = job.Id; StudioStorage.Save(DraftPath(projectId, draftId), draft); }
        finally { projectGate.Release(); }
        return job;
    }
}
