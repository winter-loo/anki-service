## prepare
This repo has three Cloud Build configs:

- `cloudbuild.base.yaml`: builds the heavy base image (`docker/Dockerfile.anki-pylib`). Run this once (or very rarely).
- `cloudbuild.web.yaml`: builds only the service image (`docker/Dockerfile.web-api`) on top of a prebuilt base image.
- `cloudbuild.cloudrun.yaml`: builds the service image and deploys to Cloud Run (intended for GitHub triggers).

In these files, `_REPOSITORY` is the Artifact Registry repository name (the container repo), not a Git repo.

It's the middle path segment in the image URL:

<LOCATION>-docker.pkg.dev/<PROJECT_ID>/<REPOSITORY>/<IMAGE>:<TAG>

With `_REPOSITORY` set to `anki`, images go to:

- asia-northeast1-docker.pkg.dev/$PROJECT_ID/anki/anki-pylib:<tag>
- asia-northeast1-docker.pkg.dev/$PROJECT_ID/anki/anki-web-api:<tag>

How to see what repositories you have:

```bash
gcloud artifacts repositories list --location asia-northeast1
```

How to create one (example repo name anki):

```bash
gcloud artifacts repositories create anki \
  --repository-format=docker \
  --location asia-northeast1
```


## build
Build the base image once (optional, but required before the first service build):

```bash
gcloud builds submit --config cloudbuild.base.yaml .
```

Then build the web API image (this is the only thing you rebuild on code changes):

```bash
gcloud builds submit --config cloudbuild.web.yaml \
  --substitutions=_ANKI_PYLIB_IMAGE=asia-northeast1-docker.pkg.dev/$PROJECT_ID/anki/anki-pylib:base,_UI_VERSION=latest \
  .
```

Tip: for true immutability, set `_ANKI_PYLIB_IMAGE` to a digest (`.../anki-pylib@sha256:...`) instead of a tag.

DOCKER_IMG=asia-northeast1-docker.pkg.dev/gen-lang-client-0344850940/anki/anki-web-api:576b4d76-9b3e-4f35-948e-5d6f7a8090e4

```bash
gcloud run deploy anki-service --image ${DOCKER_IMG} --region asia-northeast1
```

To test the image locally first:
```bash
docker pull ${DOCKER_IMG}
```
