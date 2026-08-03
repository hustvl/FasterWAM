<div align="center">

<h2><nobr>Faster-WAM: Efficient Inference-Time Future Conditioning</nobr><br><nobr>for Robust World Action Models</nobr></h2>

<b>Weiheng Zhao</b><sup>1</sup> &middot; <b>Haoyi Jiang</b><sup>1</sup> &middot; <b>Xin Shi</b><sup>2</sup> &middot; <b>Liu Liu</b><sup>3</sup> &middot; <b>Zhizhong Su</b><sup>3</sup> &middot; <b>Wei Sui</b><sup>2</sup> &middot; <b>Fan Huang</b><sup>4</sup> &middot; <b>Xinggang Wang</b><sup>1</sup>

Huazhong University of Science and Technology<sup>1</sup> &middot; D-Robotics<sup>2</sup> &middot; Horizon Robotics<sup>3</sup> &middot; Xiamen University<sup>4</sup>

</div>

The key insight behind **Faster-WAM** is that future representations are not merely an auxiliary training signal, but essential inference-time context for robust action prediction under distribution shifts. Guided by this principle, Faster-WAM computes future representations once and selectively reuses them during action denoising, reducing redundant video-action interaction. It achieves state-of-the-art in-distribution performance and robust OOD generalization across simulated and real-world manipulation, while substantially reducing inference latency.

<div align="center">
  <img src="assets/framework_r.png" alt="Faster-WAM framework" width="85%">
</div>

---

## Table of Contents

- [Release](#release)

## Release

- [ ] Training and inference code
- [ ] Model checkpoints
