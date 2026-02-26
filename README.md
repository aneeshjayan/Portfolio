# Aneesh Jayan Prabhu - Portfolio Website

A modern, interactive portfolio website showcasing AI/ML engineering expertise, projects, and experience.

## 🌟 Features

### Design
- **Modern Dark Theme** with light mode toggle
- **Glassmorphism Effects** for premium aesthetic
- **Navy Blue & Cyan Color Scheme** for professional look
- **Smooth Animations** and transitions throughout
- **Fully Responsive** design (mobile-first approach)

### Sections
1. **Hero Section**
   - Professional headshot
   - Animated typing effect showing skills
   - CTA buttons for projects, resume, and AI assistant
   - Social media links

2. **About Me**
   - Comprehensive introduction
   - Animated statistics counters
   - Interest tags
   - Key achievements

3. **Experience Timeline**
   - Interactive expandable cards
   - Wolters Kluwer internship details
   - VIT Research experience
   - Technology tags for each role

4. **Projects Showcase**
   - Filterable project grid
   - TrustMedAI, Adaptive Inference System, Optimal-SLM
   - Detailed metrics for each project
   - Modal views with comprehensive information

5. **Skills**
   - Tabbed interface (Technical Skills, Domain Expertise, Tools)
   - Interactive skill items
   - Comprehensive technology coverage

6. **Education**
   - Arizona State University (MS)
   - VIT (B.Tech)
   - Coursework and achievements

7. **Contact**
   - Contact form
   - Contact information cards
   - Availability status
   - Social media links

8. **AI Assistant Chatbot** 🤖
   - Floating chat bubble
   - Intelligent responses about projects, experience, skills
   - Quick action buttons
   - Knowledge base covering all portfolio content

## 🚀 Getting Started

### Prerequisites
- A modern web browser (Chrome, Firefox, Edge, Safari)
- Python (for local server) or any HTTP server

### Running Locally

#### Option 1: Python HTTP Server
```bash
# Navigate to the portfolio directory
cd d:\Portfolio

# Start Python HTTP server
python -m http.server 8000
```

Then open your browser and navigate to: `http://localhost:8000`

#### Option 2: Live Server (VS Code)
1. Install the "Live Server" extension in VS Code
2. Right-click on `index.html`
3. Select "Open with Live Server"

#### Option 3: Direct File Access
Simply double-click `index.html` to open in your default browser.

## 📁 File Structure

```
d:\Portfolio\
├── index.html          # Main HTML structure
├── styles.css          # Comprehensive CSS with design system
├── script.js           # JavaScript functionality and AI chatbot
├── README.md           # This file
└── [headshot image]    # Professional headshot
```

## 🎨 Customization

### Colors
Edit CSS variables in `styles.css`:
```css
:root {
  --primary-navy: #1e3a8a;
  --primary-cyan: #06b6d4;
  --dark-bg: #0a0e27;
  /* ... more variables */
}
```

### Content
- **Personal Information**: Update in `index.html`
- **Projects**: Modify project cards and modal content
- **AI Chatbot Knowledge**: Edit `aiKnowledge` object in `script.js`

### Images
Replace the headshot image path in `index.html`:
```html
<img src="path/to/your/image.png" alt="Your Name" class="hero-image">
```

## 🤖 AI Assistant

The AI chatbot can answer questions about:
- **Projects**: TrustMedAI, Adaptive Inference System, Optimal-SLM
- **Experience**: Wolters Kluwer, VIT Research
- **Skills**: Technical skills, domain expertise, tools
- **Education**: ASU, VIT
- **Contact**: How to get in touch

### Extending the AI Knowledge Base
Edit the `aiKnowledge` object in `script.js` to add more information or modify responses.

## 🌐 Deployment

### GitHub Pages
1. Create a GitHub repository
2. Push your portfolio files
3. Go to Settings > Pages
4. Select main branch as source
5. Your site will be live at `https://yourusername.github.io/repository-name`

### Netlify
1. Drag and drop the portfolio folder to Netlify
2. Your site will be live instantly

### Vercel
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel
```

## 📱 Browser Compatibility

- ✅ Chrome (latest)
- ✅ Firefox (latest)
- ✅ Safari (latest)
- ✅ Edge (latest)
- ✅ Mobile browsers (iOS Safari, Chrome Mobile)

## 🎯 Key Technologies

- **HTML5**: Semantic markup
- **CSS3**: Modern styling with custom properties, flexbox, grid
- **JavaScript (ES6+)**: Interactive functionality
- **Google Fonts**: Inter, Space Grotesk
- **No frameworks**: Pure vanilla code for maximum performance

## 📊 Performance

- Fast load times (< 2s)
- Smooth 60fps animations
- Optimized images
- Minimal dependencies

## 📄 License

This portfolio template is free to use for personal purposes. Please customize it with your own information.

## 📧 Contact

**Aneesh Jayan Prabhu**
- Email: aneeshjayan11@gmail.com
- Phone: (602) 768-6622
- LinkedIn: [linkedin.com/in/aneeshjayan](https://linkedin.com/in/aneeshjayan)
- GitHub: [github.com/aneeshjayan](https://github.com/aneeshjayan)

---

Built with ❤️ using modern web technologies
