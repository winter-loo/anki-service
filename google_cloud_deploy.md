## prepare
In `cloudbuild.yaml`, `_REPOSITORY` is the Artifact Registry repository name (the container repo), not a Git repo.

It’s the middle path segment in the image URL:

<LOCATION>-docker.pkg.dev/<PROJECT_ID>/<REPOSITORY>/<IMAGE>:<TAG>

In the file I added, _REPOSITORY is set to "anki", so images go to:

- us-central1-docker.pkg.dev/$PROJECT_ID/anki/anki-pylib:$BUILD_ID
- us-central1-docker.pkg.dev/$PROJECT_ID/anki/anki-web-api:$BUILD_ID

How to see what repositories you have:

```bash
gcloud artifacts repositories list --location asia-northeast1
```

  How to create one (example repo name anki):

```bash
gcloud artifacts repositories create anki \
  --repository-format=docker \
  --location us-central1
```


## build
I used the `gcloud` CLI to submit the build. Specifically, I ran this command:

```bash
gcloud builds submit --config cloudbuild.yaml .
```

### Breakdown of what happened:
1. **Source Upload:** It packaged the files in current directory into a tarball and uploaded them to a temporary Google Cloud Storage bucket.
2. **Build Execution:** Google Cloud Build picked up that source and followed the steps in your `cloudbuild.yaml`.
3. **Artifacts:** It built the `anki-pylib` and `anki-web-api` images and pushed them to your Artifact Registry in `asia-northeast1`.

You can always check the history of your builds by running `gcloud builds list`.
For rebuiling the image, just run the above command again.
A few things to note:
    - New Tags: Each time you run this, Google Cloud Build generates a new BUILD_ID, so your images in the Artifact Registry will have a unique tag (e.g., anki-web-api:576b4d...).
    - Cache: Cloud Build will try to reuse layers from previous builds to speed things up, but since your cloudbuild.yaml defines a multi-stage process (pylib → web-api),
      it will ensure the final image is consistent with your current code.
    - Verification: You can monitor the new build's progress by running gcloud builds list --limit 5

## deploy

DOCKER_IMG=asia-northeast1-docker.pkg.dev/gen-lang-client-0344850940/anki/anki-web-api:576b4d76-9b3e-4f35-948e-5d6f7a8090e4

To deploy to Cloud Run, run:
```bash
gcloud run deploy anki-service --image ${DOCKER_IMG} --region asia-northeast1
```

To test the image locally first:
```bash
docker pull ${DOCKER_IMG}
```
