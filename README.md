# 🚀 GitHub Pages Deployment Instructions

## Quick Fix for Your 404 Error

Follow these steps to deploy your AML Neural Monitor to GitHub Pages:

## Step 1: Repository Setup

1. **Go to your repository**: https://github.com/arjunlaxman/aml-neural-monitor
2. **Create the repository** if it doesn't exist yet:
   - Click "New" on GitHub
   - Name it: `aml-neural-monitor`
   - Make it Public
   - Don't initialize with README (we'll add our own)

## Step 2: Upload the Files

### Option A: Using GitHub Web Interface (Easiest)

1. Go to your repository
2. Click "Add file" → "Upload files"
3. Upload these files:
   - `index.html` (the main file I created)
   - `README-enhanced.md` (rename to `README.md`)
4. Commit the changes

### Option B: Using Git Command Line

```bash
# Clone your repository (if not already done)
git clone https://github.com/arjunlaxman/aml-neural-monitor.git
cd aml-neural-monitor

# Add the files
# Copy the index.html file to your repository folder
# Copy README-enhanced.md as README.md

# Commit and push
git add .
git commit -m "Add AML Neural Monitor with enhanced UI"
git push origin main
```

## Step 3: Enable GitHub Pages

1. **Go to Settings** in your repository
2. **Scroll down to "Pages"** section (in the left sidebar)
3. **Source**: Select "Deploy from a branch"
4. **Branch**: Select `main` (or `master` if that's your default)
5. **Folder**: Select `/ (root)`
6. **Click "Save"**

## Step 4: Wait and Access

1. GitHub Pages takes 2-10 minutes to deploy
2. Your site will be available at:
   ```
   https://arjunlaxman.github.io/aml-neural-monitor/
   ```
3. You can check deployment status in the Actions tab

## Step 5: Verify Deployment

Once deployed, you should see:
- ✅ A loading screen with spinning animation
- ✅ The main dashboard with dark theme
- ✅ Interactive anomaly detection cards
- ✅ Functioning "Run Analysis" button
- ✅ Export Report functionality

## 📁 File Structure

Your repository should have:
```
aml-neural-monitor/
├── index.html          # Main application (all-in-one)
├── README.md           # Project documentation
└── .gitignore          # (optional)
```

## 🔧 Troubleshooting

### Still seeing 404?

1. **Check repository name**: Must be exactly `aml-neural-monitor`
2. **Check GitHub Pages is enabled**: Settings → Pages
3. **Wait longer**: Initial deployment can take up to 20 minutes
4. **Check branch**: Ensure you're using the correct branch (main/master)
5. **Clear browser cache**: Ctrl+F5 or Cmd+Shift+R

### Page loads but looks broken?

- The single `index.html` file contains everything needed
- No external dependencies required (uses CDN links)
- Check browser console for any errors

### Want to update content?

1. Edit the `index.html` file
2. Commit and push changes
3. GitHub Pages auto-updates in 2-5 minutes

## 🎨 Customization

To customize the app:

1. **Change colors**: Edit the Tailwind classes in index.html
2. **Add more anomalies**: Modify the `detectAnomalies` function
3. **Update entity names**: Edit the `generateTransactionGraph` function
4. **Change risk scores**: Adjust values in the entity objects

## 📱 Mobile Responsive

The app is fully responsive and works on:
- 📱 Mobile phones
- 📱 Tablets
- 💻 Laptops
- 🖥️ Desktop monitors

## 🔗 Alternative: Using GitHub.io Repository

If you prefer, create a repository named `arjunlaxman.github.io`:
1. Create repository: `arjunlaxman.github.io`
2. Add the files to a subfolder: `/aml-neural-monitor/`
3. Access at: `https://arjunlaxman.github.io/aml-neural-monitor/`

## ✅ Success Checklist

- [ ] Repository created
- [ ] Files uploaded (index.html, README.md)
- [ ] GitHub Pages enabled
- [ ] Site accessible at the URL
- [ ] All features working

## 🆘 Need Help?

1. Check GitHub Pages documentation: https://pages.github.com/
2. Check GitHub status: https://www.githubstatus.com/
3. Try incognito/private browsing mode
4. Contact: arjunlaxmand40@gmail.com

---

**Note**: The enhanced version is a single-file application that works immediately when deployed. No build process or additional setup required!
