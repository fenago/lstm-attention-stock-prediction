#!/bin/bash
# Instructions for Mac users:
# Download git-push.sh
# chmod +x git-push.sh
# ./git-push.sh

echo "Running Git Push Script"
echo "=========================================="
echo ""

# Prompt for repository name
read -p "Enter GitHub repository (format: fenago/medical): " repo_name

# Validate input format
if ! echo "$repo_name" | grep -qE '^[^/]+/[^/]+$'; then
    echo "ERROR: Invalid format. Please use format: fenago/medical"
    echo "Do NOT include https://github.com/"
    read -p "Press Enter to exit..."
    exit 1
fi

# Construct full repository URL
repo_url="https://github.com/${repo_name}.git"
echo ""
echo "Repository URL: $repo_url"
echo ""

# Check if .git directory exists
if [ -d .git ]; then
    echo "Git repository already exists."
    read -p "Do you want to reinitialize it? This will keep your files but reset Git history. (Y/N): " reinit
    if [[ "${reinit,,}" == "y" ]]; then
        echo "Removing existing .git directory..."
        rm -rf .git
        echo "Initializing new Git repository with main branch..."
        git init -b main
        echo ""
    fi
else
    echo "No Git repository found."
    read -p "Initialize new Git repository? (Y/N): " first_time
    if [[ "${first_time,,}" == "y" ]]; then
        echo "Initializing new Git repository with main branch..."
        git init -b main
        echo ""
    fi
fi

echo "Setting remote repository..."
git remote remove origin 2>/dev/null
git remote add origin "$repo_url"
echo ""

echo "Checking current status..."
git status
echo ""

# Detect current branch
current_branch=$(git branch --show-current)
echo "Current branch: $current_branch"

# If on master, ask to rename to main
if [[ "${current_branch,,}" == "master" ]]; then
    echo ""
    echo "WARNING: You are on 'master' branch, not 'main'."
    read -p "Rename 'master' to 'main'? (Y/N): " rename_branch
    if [[ "${rename_branch,,}" == "y" ]]; then
        git branch -M main
        current_branch="main"
        echo "Branch renamed to main"
    fi
fi

echo ""
echo "Adding all files except those in .gitignore..."
git add .
echo ""

echo "Committing changes..."
read -p "Enter commit message (or press Enter for default message): " commit_msg
if [ -z "$commit_msg" ]; then
    git commit -m "Fixed TypeScript errors and added Google Analytics integration"
else
    git commit -m "$commit_msg"
fi

echo ""
echo "WARNING: This will FORCE PUSH to the repository, potentially overwriting remote changes."
echo "This makes the remote repository match your local repository exactly."
read -p "Are you sure you want to force push? (Y/N): " confirm

if [[ "${confirm,,}" == "y" ]]; then
    echo "Pushing to GitHub repository: $repo_url"
    git push -f -u origin "$current_branch"
else
    echo "Force push canceled."
    exit 1
fi

echo ""
echo "Done!"
echo "=========================================="
read -p "Press Enter to exit..."