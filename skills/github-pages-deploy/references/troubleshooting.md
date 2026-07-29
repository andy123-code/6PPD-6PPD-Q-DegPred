# GitHub Pages And Static Deployment Troubleshooting

## Symptom: GitHub Pages Shows README

Likely causes:
- The actual Pages source branch/folder does not contain root `index.html`.
- GitHub Pages is configured to publish from `main` root while the app exists only in `gh-pages`.
- GitHub has fallen back to rendering `README.md`.

Fix:
- Put the real app at root `index.html` on the active Pages source branch.
- Add root `.nojekyll`.
- Push to `main`.
- If source branch is uncertain, force-sync `gh-pages` to the same commit:

```bash
git push origin main
git push --force origin main:gh-pages
```

Verify:
- Open `https://<user>.github.io/<repo>/index.html?v=<short-sha>`.
- Ask user to force-refresh if they still see README.

## Symptom: Clicked Link Works But Copied Link Fails

Likely causes:
- Copied URL is missing repo path.
- Copied URL is an old Streamlit URL, not GitHub Pages.
- Browser or chat app cached old GitHub Pages output.
- URL includes a README anchor such as `#准备数据`.

Fix:
- Give a fully explicit URL:
  `https://<user>.github.io/<repo>/index.html?v=<short-sha>`
- Avoid bare `https://<user>.github.io/`.
- Avoid old Streamlit links when Streamlit visibility failed.

## Symptom: Streamlit Cloud Blank Or Login Loop

Likely causes:
- App visibility is not public.
- Streamlit auth redirects loop between app and login.
- Cloud runtime differs from local runtime.
- Server app depends on local persistence or unsupported APIs.

Fix:
- If the user needs a public, no-login link, prefer a static GitHub Pages version.
- Keep local Streamlit for long-running private work.
- In public static builds, avoid writing shared server files; use session/browser-local state and model JSON export/import.

## Symptom: Workflow Push Rejected

Message:
`refusing to allow a Personal Access Token to create or update workflow ... without workflow scope`

Cause:
- Token can push normal files but lacks `workflow` scope.

Fix:
- Ask for a new classic PAT with `repo` and `workflow`, or avoid GitHub Actions and use branch-based Pages publishing.

## Symptom: GitHub Pages API Returns 404

Likely causes:
- Token has no admin access to the repo.
- Token scopes are empty or insufficient.
- Pages endpoint is hidden for unauthorized callers.

Fix:
- Check token scopes with:

```bash
curl -sS -D - -o /dev/null -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/user
```

- If `x-oauth-scopes` is empty or missing needed scopes, do not retry API calls.
- Use browser settings or branch source fixes.
