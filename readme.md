# Using SourceTree with Git: A Basic Guide

SourceTree is a free Git GUI that makes it easy to interact with your Git repositories visually. This guide will help you understand the basics of Git and how to use it with SourceTree.

## Table of Contents

- [What is Git?](#what-is-git)
- [Installing SourceTree](#installing-sourcetree)
- [Getting Started with SourceTree](#getting-started-with-sourcetree)
- [Rules for Adding Code](#rules-for-adding-code)

## What is Git?

Git is for tracking changes in code. Multiple people can work on the same project and see who did what.

## Installing SourceTree

1. Go to the official [SourceTree website](https://www.sourcetreeapp.com/).
2. Download the appropriate version for your OS (Windows or MacOS).
3. You can skip adding Bitbucket Server step.
4. Follow the installation instructions.
5. You can click no on the message about adding an SSH key after installation.

## Setting up SourceTree with GitHub

1. **Open SourceTree** and click on 'Settings' or 'Preferences'.
2. Navigate to 'Authentication'.
3. Click 'Add' and choose 'GitHub' from the dropdown.
4. Log in with your GitHub credentials.
5. For two-factor authentication, generate an access token from GitHub settings and use that.


## Getting Started with SourceTree

1. **Clone a Repository:** When you first open SourceTree, you can clone an existing Git repository by pasting its URL and specifying a local directory. It's easier if the directory is empty otherwise git will complain.
2. **View Commits:** After the repository is cloned, you can see the list of commits, the changes made in each commit, and the commit message.
3. **Make Changes:** Modify files in your local directory, and they will appear in SourceTree under 'Unstaged files'.
4. **Stage Changes:** Before committing, you need to stage the changes. Select the files you want to commit and click on 'Stage Selected'.
5. **Commit:** After it's staged, write a commit message describing your changes and click on 'Commit'.
6. **Push and Pull:** Use the 'Push' and 'Pull' buttons to synchronize your changes with the remote repository.

## Rules for Adding Code

1. **Directory Structure:** Each assignment should be in its own subfolder. Properly name each folder according to the assignment/task. For example: `assignment_1/question_1/task_1.py`, etc.
3. **Commit Messages:** Make sure your commit messages are descriptive enough. For example, instead of "changes" or "fixes", write "Changed X to fix Y" or "Fixed a bug that did X instead of Y".
4. **Testing:** Before pushing your code, make sure it runs without errors. It's a good practice to test everything locally.
5. **Documentation:** Include enough information either in the code or a separate readme.md so other people can run your code if needed.
6. **Branching*** I don't think we will be using branching and a simple directory structure for each assignment/task/subtask should be fine.
