下面给你一版**可直接用于 15–20min 线上英文 presentation 的完整稿**，结构按你确认的方向：

```text
Part 1: Why we need this
Part 2: What abstraction we propose
Part 3: How we roughly implement it in SGLang
Part 4: How we split the work into the original 3 RFC milestones
```

我会尽量写成**口语化但正式的社区汇报风格**，不是论文式文本。

---

# Presentation Script

## Title

**Internal Process-level Fault Control for SGLang**

---

## Opening

Hi everyone, thanks for giving me the time to introduce this RFC.

Today I would like to present our proposal for an internal process-level fault-control framework for SGLang.

The goal of this RFC is not to make SGLang itself a full cluster-level fault-tolerance system. Instead, the goal is more focused: when a fault happens inside the SGLang runtime, the engine should not immediately become a black box or directly exit without giving the upper-layer system a chance to observe and control it.

So the key question we want to answer is:

**How can SGLang expose structured fault state and provide safe control primitives, so that an external serving framework can make the right recovery decision?**

I will cover four parts:

First, why we think this is needed.

Second, what abstraction we propose.

Third, how we roughly implement it inside SGLang.

And finally, how we plan to split the work into milestones.

---

# Part 1: Why We Need This

Large-scale LLM serving is increasingly deployed across multiple physical nodes. This is especially true for large MoE models, where we may use DP, EP, TP, or related parallel strategies to improve throughput.

But this also means the reliability requirement becomes much higher. A local failure, for example a single failed rank, a communication timeout, or a scheduler-side exception, may directly affect the whole serving instance.

Today, there are generally two kinds of behavior.

The first one is fail-stop. When an important process fails, the whole engine exits, and the upper-layer system restarts it.

This is simple and safe, but it is also expensive. The external serving system has very limited visibility into what happened inside the engine. It only knows that the process is gone or unhealthy.

The second one is backend-level fault handling. For example, an FT-enabled backend may detect a failed rank and isolate it from the forward path. This is useful because it can prevent healthy ranks from being blocked forever.

But this also has limitations. If fault handling is embedded directly into the inference workflow, the upper-layer serving system still cannot easily observe the actual engine state or control the recovery process. The engine still behaves like a black box during failures. This limitation is also the motivation stated in the RFC: current fault handling is either direct exit or rank isolation by FT backends, but this does not provide a unified control interface for upper-layer orchestration. ([GitHub][1])

So we believe SGLang needs an engine-level fault-control layer.

The key point is not only to detect a fault. The key point is to make the engine observable and controllable after the fault.

When a fault happens, SGLang should be able to:

stop accepting new normal inference requests,

enter a structured fault state,

keep the management interface available,

expose what happened,

and wait for an explicit instruction, such as pause, retry, scale-down, or terminate.

So the first message of this RFC is:

**Fault tolerance should not only be a data-plane capability. SGLang also needs a control-plane abstraction.**

---

# Part 2: What Abstraction We Propose

The abstraction we propose is to separate fault tolerance into three layers.

The first layer is the **data plane**.

This is where low-level fault detection and communication-level handling happen. For example, FT backends such as Mooncake or NIXL can detect communication timeouts, isolate failed ranks, or maintain an active rank mask.

The second layer is the **SGLang fault-control layer**.

This is what this RFC focuses on. SGLang maintains runtime fault state, aggregates fault signals, exposes status and control APIs, freezes or resumes admission, and executes engine-local control commands.

The third layer is the **decision plane**.

This belongs to the upper-layer serving framework. It has a broader view of the cluster, SLA, multiple engines, node health, and scheduling policy. Therefore, it should decide whether the right action is retry, scale down, restart, replace a rank, or terminate.

This is also the separation described in the RFC: data plane is implemented by FT backends, control plane is implemented by the SGLang FT framework, and decision plane is implemented by the upper-layer serving framework. ([GitHub][1])

The most important design principle is:

**SGLang should provide mechanisms, not hard-code global recovery policies.**

For example, SGLang may expose that rank 3 is unhealthy, the scheduler is paused, and the old communication domain has been aborted. But SGLang should not be the component that decides whether this should be handled by retry, scale-down, or full restart.

That decision requires global context.

So the control flow becomes:

A fault happens.

SGLang brings the engine into a controlled state.

The upper-layer serving framework observes the state.

The upper-layer serving framework sends a recovery instruction.

SGLang executes the instruction safely inside the engine.

In terms of interfaces, we propose two basic APIs.

The first one is:

```http
GET /fault_tolerance/status
```

This returns the current engine-level fault state, per-rank or per-component health information, and the latest fault information.

The second one is:

```http
POST /fault_tolerance/apply
```

This is used by the upper-layer system to apply a control instruction, such as pause, retry, scale_down, or terminate.

The RFC currently lists status and control interfaces, with example control actions including `pause`, `retry`, and `scale_down`. ([GitHub][1])

So the second message of this RFC is:

**We want to standardize how SGLang reports fault state and how external systems control recovery.**

---

# Part 3: How We Roughly Implement It in SGLang

Now I will talk about the rough implementation inside SGLang.

At a high level, we introduce two core components.

The first component is **SentinelManager**.

It lives in the main process. It owns the global fault-tolerance state machine, controls admission, serves the status and apply APIs, receives fault events, and sends commands to scheduler processes.

The second component is **FaultSentinel**.

Each scheduler process has one local FaultSentinel. It is a lightweight control thread inside the scheduler process. It sends heartbeat to the SentinelManager, reports local faults, receives out-of-band commands, and performs emergency actions such as hard pause or communication abort.

So the layout is:

```text
Main Process
  - HTTP Server
  - TokenizerManager
  - SentinelManager

Scheduler Process
  - Scheduler main loop
  - ModelRunner / TpModelWorker
  - FaultSentinel control thread
```

The implementation spec follows this structure: the main process owns `SentinelManager`, and each scheduler process owns a `FaultSentinel` control thread plus the scheduler main loop wrapper. 

A natural question is: why do we need a FaultSentinel control thread?

The reason is that fault-control commands cannot depend only on the scheduler main loop.

In a normal case, the scheduler loop can process normal pause or continue requests. But in a fault case, the scheduler main loop may be blocked in model forward, collective communication, blocking I/O, or exception handling.

If a hard pause command goes through the normal scheduler queue, it may never be processed.

So the FaultSentinel provides an out-of-band control path. It is not part of the normal inference data path. Its job is to keep the process controllable even when the main execution path is unhealthy.

However, we also need to be careful about thread safety.

The FaultSentinel should not directly rebuild ModelRunner, rebind process groups, or recapture CUDA graphs. These operations are complex runtime mutations and should happen on the scheduler main thread at a safe point.

So we separate emergency actions from reconstruction.

The FaultSentinel can perform emergency control actions, such as disabling communicators or aborting old communication domains.

But reinitialization, ModelRunner rebind, runtime state cleanup, and graph recapture should be executed by the scheduler main loop when it enters a parked recovery point. This is also explicitly required in the development spec: hard pause and communication abort must use an out-of-band control channel, while reinit, ModelRunner rebind, and CUDA graph recapture should happen in the scheduler main loop at a parked safe point. 

Now let me describe the fault handling path.

When a scheduler-side fault happens, for example a Python exception, communication error, or heartbeat stall, we first wrap it into a structured `FaultEvent`.

At this stage, we do not try to decide whether the fault is recoverable.

The first goal is to bring the engine into a controlled state.

The flow is roughly:

```text
scheduler exception / communication failure / heartbeat stall
    -> FaultEvent
    -> FaultSentinel reports the event
    -> SentinelManager enters FAULT_DETECTED
    -> admission is frozen
    -> hard pause is sent to scheduler processes
    -> old communicators are aborted or destroyed
    -> engine enters COMM_ABORTED or WAITING_OPERATOR
```

The important principle is:

**Detection does not imply recoverability.**

Detection only means the fault is now under the framework. Whether retry is safe is decided later, based on process liveness, communication cleanup result, reinitialization result, and health check.

This is also the requirement in the spec: scheduler exceptions are first wrapped as `FaultEvent`; they no longer need to be pre-classified as recoverable; recoverability is judged during the recovery phase. 

Next is admission control.

When the engine is in `RUNNING` state, normal inference requests are allowed.

When the engine leaves `RUNNING`, normal inference requests should return a structured 503 response. But management APIs must remain available, including fault status, fault apply, health, metrics, and necessary admin endpoints.

This is important because if the management interface disappears, the upper-layer system cannot observe or recover the engine.

Next is hard pause and communication abort.

A normal pause is not enough for fault recovery. If one rank is blocked in collective communication, simply setting a pause flag may not stop the blocked operation.

Therefore, hard pause should include communication cleanup.

The rough flow is:

```text
freeze admission
    -> best-effort normal pause
    -> broadcast HARD_ABORT_COMM
    -> local FaultSentinel disables communicators
    -> abort or destroy old process groups
    -> enter COMM_ABORTED or WAITING_OPERATOR
```

If communication abort fails, we should not silently resume. We should keep the management plane available and expose the failure state.

Finally, the retry path.

The first implementation focuses on same-topology retry. That means we do not change world size or parallel configuration in the first phase.

The retry flow is:

```text
POST /fault_tolerance/apply retry
    -> SentinelManager validates current state
    -> prepare retry
    -> ensure old communication domain is aborted
    -> send RETRY_REINIT
    -> scheduler main loop executes recovery at parked safe point
    -> cleanup scheduler runtime state
    -> reinit distributed environment
    -> rebuild process groups
    -> rebind ModelRunner group-dependent objects
    -> invalidate or recapture CUDA graphs
    -> run health collective
    -> resume scheduler and admission
```

The key point is that retry is conservative. We only resume if the health check succeeds.

Also, we should be conservative about in-flight requests. Requests that are safe to recompute can be retracted or retried. Requests that have already emitted partial streaming output should not be silently continued as if nothing happened. They should be marked interrupted or handled by the upper-layer retry policy.

The development spec also lists the runtime state that must be cleaned before retry, including running batch, batch queue, overlap scheduling state, chunked request state, unsafe forward state, KV cache blocks, and partial streaming output state. 

So the third message of this RFC is:

**The implementation is designed to keep the engine controllable during faults, while keeping unsafe runtime reconstruction on the scheduler main thread.**

---

# Part 4: Milestones and Landing Plan

Now I will describe how we propose to land this work.

Instead of submitting a big-bang fault-tolerance system, we plan to split it into the three milestones described in the RFC.

## Milestone 1: Fault Reporting

The first milestone is fault reporting.

The goal is that when a failure happens in an inference-related component, SGLang does not immediately exit when fault tolerance is enabled. Instead, the runtime stays alive for a configurable timeout window and exposes internal fault state through newly introduced APIs.

This allows the upper-layer serving framework to observe the failure and coordinate the next step.

In this milestone, we mainly need:

the FT configuration flag,

the state model,

the `FaultEvent` structure,

the `SentinelManager` skeleton,

the status API,

basic fault recording,

and tests for state transition and fault event idempotency.

The key value of this milestone is that it introduces observability without changing recovery behavior too much.

It also gives the community a chance to review the API shape and the state model early.

The RFC defines Milestone 1 exactly in this direction: after a failure, the runtime no longer exits immediately, but stays alive for a configurable timeout window and exposes internal fault state through APIs. ([GitHub][1])

## Milestone 2: Pause-on-error

The second milestone is pause-on-error.

Once a fault is reported, healthy components should not continue running blindly. Otherwise, we may get cascading failures, blocked collectives, or corrupted runtime state.

So in this milestone, we introduce the pause instruction and the out-of-band control path.

The SentinelManager freezes admission and broadcasts pause to local FaultSentinels.

Each FaultSentinel sets local pause state and helps stop the workflow in a controlled way.

For hard pause, we also need communication cleanup, including communicator disable, abort, or process group destroy.

This milestone makes the engine controllable after fault detection.

The RFC describes this milestone as introducing a `Pause` instruction to suspend healthy ranks when one rank fails, so that cascading failures can be prevented. Sentinels set a pause flag, and key components such as scheduler and model runner check this flag to halt the workflow in a controlled manner. ([GitHub][1])

In implementation, we will be conservative.

The FaultSentinel can handle emergency control actions.

But it will not directly mutate complex runtime objects like ModelRunner or CUDA graphs.

Those operations remain on the scheduler main thread.

## Milestone 3: Fault Handling Interface

The third milestone is the fault handling interface.

Based on fault reporting and pause-on-error, the upper-layer serving framework can choose a recovery strategy.

The first strategy we plan to support is same-topology retry.

This includes:

validating the current state,

cleaning old communication domains,

cleaning unsafe scheduler runtime state,

reinitializing distributed environment with the same topology,

rebinding ModelRunner and group-dependent objects,

invalidating or recapturing CUDA graphs,

running a health collective,

and resuming admission only after success.

In the future, the same interface can support scale-down or other recovery actions. But we do not need to solve all of them in the first implementation.

The RFC defines Milestone 3 as adding the fault handling interface, so that the upper-layer serving framework can choose recovery strategies such as retry and scale_down, and the runtime can restore context or isolate failed components if needed. ([GitHub][1])

So the landing plan is:

Milestone 1 gives observability.

Milestone 2 gives controlled pause.

Milestone 3 gives recovery command execution.

And all of this should be disabled by default, so existing SGLang behavior remains unchanged unless users explicitly enable the feature.

---

# Closing

To summarize:

This RFC is not trying to make SGLang a full cluster-level fault-tolerance decision system.

Instead, it adds an engine-level fault-control layer.

The data plane detects or reports faults.

SGLang maintains structured state and executes engine-local commands.

The upper-layer serving framework makes the recovery decision.

The first implementation focuses on three incremental milestones:

fault reporting,

pause-on-error,

and fault handling interface.

We believe this gives SGLang a clean and extensible foundation for future fault tolerance, while keeping the initial implementation incremental, opt-in, and reviewable.

What we would like to get from the community today is feedback on three things:

First, whether this abstraction boundary is acceptable.

Second, whether the proposed status and control APIs are the right interface for upper-layer orchestration.

And third, whether the three milestone plan is a reasonable way to land this feature incrementally.

Thank you.

---

# 10 Key Q&A

## Q1: Why should this logic live inside SGLang instead of only in the serving platform?

**Answer:**

The serving platform should make the recovery decision, but it cannot safely perform engine-local operations from outside the process.

For example, freezing admission, pausing scheduler workflow, aborting old communicators, rebuilding process groups, rebinding ModelRunner objects, and invalidating CUDA graphs are all runtime-internal operations.

So the split is:

SGLang provides engine-local mechanisms.

The serving platform provides global recovery policy.

That is exactly the abstraction we propose.

---

## Q2: Is this duplicating FT backends such as Mooncake or NIXL?

**Answer:**

No. FT backends and this RFC work at different layers.

FT backends are data-plane components. They can detect low-level communication failures and may isolate failed ranks.

This RFC adds the control plane above that. It aggregates fault signals, exposes engine state, freezes admission, and executes explicit recovery commands.

So we are not replacing FT backends. We are making their fault signals observable and controllable at the engine level.

---

## Q3: Why not just fail-stop and let Kubernetes or another supervisor restart the engine?

**Answer:**

Fail-stop should remain the default and the safest fallback.

But it is not always the cheapest recovery path.

For some failures, such as transient communication issues or recoverable scheduler-side failures, the process may still be alive and controllable. In those cases, a controlled pause and same-topology retry may recover faster than a full restart.

This RFC does not remove restart. It adds another option before restart.

---

## Q4: Why do we need a FaultSentinel thread? Why not use the scheduler loop?

**Answer:**

Because in a fault case, the scheduler loop may be exactly the component that is blocked.

It may be blocked in model forward, collective communication, blocking I/O, or exception handling.

If hard pause depends on the normal scheduler queue, the command may never be processed.

The FaultSentinel gives us an out-of-band control path, so the process remains controllable even when the main execution path is unhealthy.

---

## Q5: Is it safe to abort communicators from a separate thread?

**Answer:**

This is one of the sensitive parts, so the design is conservative.

The FaultSentinel thread is only allowed to perform emergency control actions, such as disabling or aborting communicators with timeout.

It should not rebuild ModelRunner, mutate scheduler runtime state, or recapture CUDA graphs.

Those operations are executed later by the scheduler main thread at a parked safe point.

If communication cleanup fails, we do not resume. We keep the engine in a controlled fault state and expose the failure through the status API.

---

## Q6: How do you decide whether a fault is recoverable?

**Answer:**

We do not decide that at detection time.

At detection time, all scheduler-side faults are first wrapped into structured `FaultEvent`s and brought into the framework.

Recoverability is judged later during recovery, based on whether the relevant processes are alive, whether old communicators can be cleaned up, whether distributed reinit succeeds, and whether the health collective passes.

So detection and recovery are intentionally separated.

---

## Q7: What happens to in-flight requests when a fault happens?

**Answer:**

We should be conservative.

The first goal is to recover engine availability, not to guarantee transparent continuation for every in-flight request.

Requests that can be safely retracted or recomputed may be retried.

Requests that have already emitted partial streaming output should not be silently continued as if nothing happened. They should be marked interrupted or handled by the upper-layer retry policy.

This avoids correctness issues caused by inconsistent runtime state after a fault.

---

## Q8: Why is the first recovery target same-topology retry instead of scale-down?

**Answer:**

Same-topology retry is the smallest complete recovery loop.

It validates the whole framework: fault reporting, admission freeze, pause, communication cleanup, distributed reinit, health check, and resume.

Scale-down is important, but it requires topology change, rank isolation semantics, request redistribution, and possibly expert migration.

So we propose to first land same-topology retry, then reuse the same control framework for scale-down later.

---

## Q9: What happens if retry fails?

**Answer:**

Retry failure should not make the management plane disappear.

If retry fails, the engine should stay in a structured fault state, such as `COMM_ABORTED` or `WAITING_OPERATOR`.

The status API should expose which stage failed, which component or rank failed, and the error summary.

Then the upper-layer serving framework can decide whether to terminate, restart, or apply another recovery strategy.

---

## Q10: How do we avoid making SGLang too complex?

**Answer:**

We control complexity in three ways.

First, the feature is disabled by default, so existing behavior remains unchanged.

Second, the work is split into three milestones: fault reporting, pause-on-error, and fault handling interface.

Third, the abstraction keeps policy outside SGLang. SGLang only provides engine-local mechanisms and APIs.

This makes the design incremental, reviewable, and extensible.

[1]: https://github.com/sgl-project/sglang/issues/22344 "[RFC]: Internal Process-level Fault Tolerance for SGLang · Issue #22344 · sgl-project/sglang · GitHub"
