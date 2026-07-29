---
name: github-pages-deploy
description: Deploy and troubleshoot static websites on GitHub Pages, especially when a public link shows README instead of the app, renders a blank page, behaves differently when copied to another browser, has branch/source confusion, cache issues, or token/API failures. Use for GitHub Pages, static HTML apps, Streamlit-to-static fallbacks, public share links, and diagnosing GitHub Pages deployment state.
---

# GitHub Pages Deploy

## Core Workflow

1. Identify the desired public URL and the expected app entrypoint.
   - Project pages use `https://<user>.github.io/<repo>/`.
   - User pages use `https://<user>.github.io/`.
   - Prefer an explicit `index.html` URL while debugging.

2. Inspect the repository source branch that GitHub Pages is actually using.
   - Check repo Settings -> Pages when browser access is available.
   - If settings are unavailable, make both likely sources valid: root of `main` and root of `gh-pages`.

3. Ensure the published branch root contains:
   - `index.html` as the real app entry.
   - `.nojekyll` to disable Jekyll processing.
   - No dependency on server-side Python, local files, Streamlit sessions, or secrets for the public static app.

4. Push changes and verify with a cache-busted URL:
   - `https://<user>.github.io/<repo>/index.html?v=<commit>`
   - If the page still shows old content, wait 1-5 minutes and force-refresh.

5. If GitHub Pages shows README instead of the app, treat it as a source/entrypoint bug:
   - Root `index.html` is missing from the actual Pages source branch, or Pages is publishing a different branch/folder than expected.
   - Put the app at root `index.html` on the selected branch.
   - If uncertain, also force-sync `gh-pages` to the same fixed commit.

## Commands

Use these commands from the repo root after checking the worktree:

```bash
git status --short --branch
git add index.html .nojekyll
git commit -m "Serve static app from Pages root"
git push origin main
git push --force origin main:gh-pages
```

Use `--force` only for a generated `gh-pages` deployment branch when it does not contain user-authored source work.

## Streamlit Fallback Rule

If a Streamlit Community Cloud app is blank, private, or loops through auth/login and the user wants a link anyone can open, do not keep fighting platform visibility. Convert the experience to a static browser app when feasible:

- CSV parsing and simple ML can run in JavaScript in the browser.
- User data should stay local in the browser.
- Provide model export/import as JSON for persistence.
- Host on GitHub Pages.

Keep the Python/Streamlit app in the repository for local or advanced use, but publish the static app for sharing.

## Token And Permissions

- `git push` can work with a token that cannot modify workflows or Pages settings.
- Creating/updating `.github/workflows/*.yml` requires a classic PAT with `workflow` scope.
- GitHub Pages API/settings changes may require repo admin access.
- If `x-oauth-scopes` is empty or missing needed scopes, do not keep retrying API calls; switch to browser settings or branch-based fixes.

## Diagnostics

Read `references/troubleshooting.md` when:

- The copied link and clicked link behave differently.
- GitHub Pages displays README.
- The page is blank but headers/toolbars appear.
- Curl/browser shows redirects, 404, or old content.
- GitHub rejects workflow pushes or API Pages changes.
