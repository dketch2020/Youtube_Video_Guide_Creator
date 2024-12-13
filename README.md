# YouTube Video Guide Generator

## Overview

This application allows you to download YouTube videos, transcribe the audio, correct punctuation using OpenAI's ChatGPT API, and generate a formatted Word document with screenshots.

## Setup Instructions

### 1. Prerequisites

- **ffmpeg:** Bundled with the application. No additional installation required.
- **OpenAI API Key:** Required for punctuation correction.

### 2. Configuration

#### a. Editing `config.json`

1. **Navigate to the Application Directory:**

   - **Windows:** Open `dist\your_script\` folder.
   - **macOS/Linux:** Open `dist/your_script/` folder.

2. **Open `config.json`:**

   - Use a text editor (e.g., Notepad, TextEdit, VS Code) to open `config.json`.

3. **Update Configuration:**

   ```json
   {
       "OPENAI_API_KEY": "your_actual_api_key_here",
       "SAVE_DIRECTORY": "C:/Users/YourUsername/Documents/YouTubeGuides"
   }