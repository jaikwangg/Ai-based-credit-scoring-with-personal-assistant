# GitHub Setup & Deployment Guide

Quick guide to set up your repository and deploy the Credit Scoring app.

## Initial GitHub Setup

### 1. Initialize Git Repository (if not already done)

```bash
git init
git add .
git commit -m "Initial commit: Credit Scoring app"
```

### 2. Create GitHub Repository

1. Go to [github.com](https://github.com) and create a new repository
2. **Don't** initialize with README (you already have one)
3. Copy the repository URL

### 3. Connect Local Repository to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

## Deployment Options

### Option A: Vercel (Recommended - Easiest)

#### Quick Deploy via Dashboard:
1. Go to [vercel.com](https://vercel.com)
2. Sign in with GitHub
3. Click "Add New Project"
4. Import your repository
5. Click "Deploy" (Vercel auto-detects Next.js + Bun)

#### Automated Deploy via GitHub Actions:
1. **Get Vercel credentials:**
   ```bash
   # Install Vercel CLI (optional, for getting IDs)
   bun add -g vercel
   vercel login
   ```

2. **Add GitHub Secrets:**
   - Go to your GitHub repo → Settings → Secrets and variables → Actions
   - Add these secrets:
     - `VERCEL_TOKEN`: Get from [vercel.com/account/tokens](https://vercel.com/account/tokens)
     - `VERCEL_ORG_ID`: Found in Vercel project settings
     - `VERCEL_PROJECT_ID`: Found in Vercel project settings

3. **Push to main branch:**
   ```bash
   git push origin main
   ```
   - The workflow will automatically deploy!

### Option B: GitHub Pages (Static Site)

1. **Enable GitHub Pages:**
   - Go to repo Settings → Pages
   - Source: GitHub Actions

2. **Push to main:**
   ```bash
   git push origin main
   ```
   - The workflow will build and deploy automatically
   - Your site will be at: `https://YOUR_USERNAME.github.io/YOUR_REPO_NAME`

### Option C: Manual Deployment

Build locally:
```bash
bun install
bun run build
bun run start
```

## GitHub Actions Workflows

Three workflows are included:

1. **CI** (`.github/workflows/ci.yml`)
   - Runs on every push/PR
   - Tests build and linting
   - No setup required

2. **Deploy to Vercel** (`.github/workflows/deploy.yml`)
   - Auto-deploys on push to main/master
   - Requires Vercel secrets (see Option A above)

3. **Deploy to GitHub Pages** (`.github/workflows/deploy-static.yml`)
   - Auto-deploys static site on push to main/master
   - Requires GitHub Pages enabled

## Verify Deployment

1. **Check GitHub Actions:**
   - Go to your repo → Actions tab
   - See workflow runs and status

2. **Check Vercel:**
   - Go to vercel.com dashboard
   - See deployment status and URL

3. **Check GitHub Pages:**
   - Go to repo Settings → Pages
   - See deployment status and URL

## Troubleshooting

### Workflow fails with "Bun not found"
- The workflow uses `oven-sh/setup-bun@v1` which should handle this
- Check workflow logs for specific errors

### Vercel deployment fails
- Verify all three secrets are set correctly
- Check Vercel project exists
- Ensure Vercel token has correct permissions

### GitHub Pages not updating
- Check if GitHub Pages is enabled
- Verify workflow ran successfully
- Check Actions tab for errors

## Next Steps

1. ✅ Push code to GitHub
2. ✅ Choose deployment platform (Vercel recommended)
3. ✅ Set up secrets (if using Vercel)
4. ✅ Push to main branch
5. ✅ Verify deployment
6. ✅ Share your live app! 🎉

## Need Help?

- Check [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed deployment options
- Check [.github/workflows/README.md](./.github/workflows/README.md) for workflow details
- Check [README.md](./README.md) for general project info
