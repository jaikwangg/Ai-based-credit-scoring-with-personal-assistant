# GitHub Actions Workflows

This directory contains GitHub Actions workflows for CI/CD.

## Workflows

### 1. `ci.yml` - Continuous Integration
- **Triggers**: Push and Pull Requests to main/master/develop
- **Purpose**: Run tests, linting, and build verification
- **No secrets required**

### 2. `deploy.yml` - Deploy to Vercel
- **Triggers**: Push to main/master
- **Purpose**: Automatically deploy to Vercel production
- **Required Secrets**:
  - `VERCEL_TOKEN`: Vercel authentication token
  - `VERCEL_ORG_ID`: Vercel organization ID
  - `VERCEL_PROJECT_ID`: Vercel project ID

**How to get Vercel credentials:**
1. Go to [vercel.com/account/tokens](https://vercel.com/account/tokens) → Create token
2. Go to your Vercel project → Settings → General → Copy Org ID and Project ID
3. Add these as secrets in GitHub: Settings → Secrets and variables → Actions

### 3. `deploy-static.yml` - Deploy to GitHub Pages
- **Triggers**: Push to main/master
- **Purpose**: Build static site and deploy to GitHub Pages
- **Required**: GitHub Pages must be enabled in repository settings
- **No additional secrets required** (uses `GITHUB_TOKEN`)

## Setup Instructions

1. **Enable GitHub Actions:**
   - Go to repository Settings → Actions → General
   - Enable "Allow all actions and reusable workflows"

2. **For Vercel deployment:**
   - Add the three Vercel secrets mentioned above
   - Push to main/master to trigger deployment

3. **For GitHub Pages:**
   - Go to Settings → Pages
   - Source: GitHub Actions
   - Push to main/master to trigger deployment

## Workflow Status

Check workflow status in the "Actions" tab of your GitHub repository.
