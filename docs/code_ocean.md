# Preparing a Code Ocean capsule

This repository includes a data and historical-prediction audit. It does not yet include a verified Code Ocean run script or a fully reproduced training environment. Capsule preparation must state exactly which analysis can be run successfully.

## Resolve OSL access first

An **Access Denied** message when creating a capsule can be an account eligibility issue. Creation and execution in the Open Science Library are available through participating publishers and collaborating institutions. Follow the institution or journal invitation, or contact `support@codeocean.com` with publication details and the access error. An ordinary account or a GitHub repository alone does not establish eligibility. See the [official OSL access guidance](https://docs.codeocean.com/osl-guide/getting-started/who-can-use-the-open-science-library).

Provide the journal, article title, manuscript reference, account email and invitation status privately to support. Confirm account activation, publication association and applicable compute/storage limits. These publication-specific details do not belong in this public repository.

## Prepare and verify the capsule

1. Confirm eligibility and the correct account through the journal or support.
2. Import the reviewed repository version using **Add Capsule → copy from public git**, then record the imported commit. Importing copies the repository; do not assume later GitHub changes are automatically synchronized. Follow the [Git import instructions](https://docs.codeocean.com/osl-guide/version-control/moving-repositories-in-and-out-of-code-ocean/importing-git-repositories).
3. Build the required environment, arrange code/data/weights and configure a master run script. Remove machine-specific paths and write generated outputs to `/results`. Confirm distribution rights and acquisition instructions for third-party artifacts.
4. Run the declared analysis through a **Reproducible Run**, inspect its outputs and record resource use. The current audit can support an explicitly scoped historical-data check; it is not a substitute for fixed-checkpoint inference or full retraining. Review the [run-script guidance](https://docs.codeocean.com/osl-guide/user-manual/reproducible-code-execution/how-to-write-and-set-a-run-script).
5. Complete metadata and confirm the intended submission route with the editor. In an integrated journal workflow, associate the article in the capsule metadata and check the peer-review badge and submission mode before submitting. The [peer-review documentation](https://docs.codeocean.com/osl-guide/publishing-on-code-ocean/peer-review) distinguishes private integrated review from non-integrated publication; do not assume every submission is private.

Before claiming complete reproduction, resolve the data, target, fold, validation and checkpoint limitations in the [reproducibility guide](../reproducibility/README.md). Code Ocean execution verification does not establish that the experimental design or scientific interpretation is correct.

Official documentation checked on 8 September 2026. No capsule creation, upload or submission is performed by the repository audit.
