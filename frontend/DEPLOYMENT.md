# Deployment Guide

This guide covers different deployment options for the Credit Scoring app.

## Quick Deploy to Vercel

### Method 1: Vercel Dashboard (Easiest)

1. Push your code to GitHub
2. Go to [vercel.com](https://vercel.com) and sign in
3. Click "Add New Project"
4. Import your GitHub repository
5. Vercel will auto-detect:
   - Framework: Next.js
   - Build Command: `bun run build`
   - Output Directory: `.next`
   - Install Command: `bun install`
6. Click "Deploy"

Your app will be live in minutes!

### Method 2: Vercel CLI

```bash
# Install Vercel CLI
bun add -g vercel

# Login
vercel login

# Deploy
vercel

# Deploy to production
vercel --prod
```

### Method 3: GitHub Actions (Automated)

1. **Get Vercel credentials:**
   - Go to [Vercel Settings > Tokens](https://vercel.com/account/tokens)
   - Create a new token (copy it)
   - Go to your project settings in Vercel
   - Copy `Org ID` and `Project ID`

2. **Add GitHub Secrets:**
   - Go to your GitHub repository
   - Settings > Secrets and variables > Actions
   - Add these secrets:
     - `VERCEL_TOKEN`: Your Vercel token
     - `VERCEL_ORG_ID`: Your organization ID
     - `VERCEL_PROJECT_ID`: Your project ID

3. **Push to main/master branch:**
   - The workflow will automatically deploy on every push

## Deploy to GitHub Pages

1. **Enable GitHub Pages:**
   - Go to repository Settings > Pages
   - Source: GitHub Actions

2. **Push to main/master:**
   - The workflow `.github/workflows/deploy-static.yml` will:
     - Build the static site
     - Deploy to GitHub Pages
   - Your site will be available at: `https://[username].github.io/[repo-name]`

**Note:** GitHub Pages requires static export. The workflow handles this automatically.

## Deploy to Other Platforms

### Netlify

1. Connect your GitHub repository to Netlify
2. Build settings:
   - Build command: `bun run build`
   - Publish directory: `.next`
   - Install command: `bun install`

### Railway

1. Connect your GitHub repository
2. Railway will auto-detect Next.js
3. Set build command: `bun run build`
4. Set start command: `bun run start`

### Docker

Create a `Dockerfile`:

```dockerfile
FROM oven/bun:latest

WORKDIR /app

COPY package.json bun.lockb* ./
RUN bun install

COPY . .
RUN bun run build

EXPOSE 3000

CMD ["bun", "run", "start"]
```

Build and run:
```bash
docker build -t credit-scoring .
docker run -p 3000:3000 credit-scoring
```

## Environment Variables

If you need environment variables:

1. **Vercel:** Add in Project Settings > Environment Variables
2. **GitHub Actions:** Add as repository secrets
3. **Local:** Create a `.env.local` file (not committed to git)

## Troubleshooting

### Build fails with Bun

- Ensure Bun is installed in the deployment environment
- Use `bun install` instead of `npm install`
- Check that `package.json` has correct scripts

### Static export issues

- Some Next.js features require server-side rendering
- If deploying to static hosting, ensure all pages are static
- Check `next.config.js` for `output: 'export'`

### Port issues

- Vercel/Netlify handle ports automatically
- For manual deployment, ensure port 3000 is available
- Use `PORT` environment variable if needed

## Monitoring

After deployment:
- Check build logs in your platform's dashboard
- Monitor error logs
- Set up analytics (optional)
- Configure custom domain (optional)
