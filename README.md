# 🧩 AML Entity Resolution — Interactive Fraud Network Visualizer (v1.0)

<div align="center">

![HTML](https://img.shields.io/badge/HTML-5-orange?style=for-the-badge&logo=html5&logoColor=white)
![CSS](https://img.shields.io/badge/CSS-3-blue?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-yellow?style=for-the-badge&logo=javascript&logoColor=black)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)

**Live Demo:** https://arjunlaxman.github.io/aml-entity-resolution/

</div>

---

## 📌 Overview

**AML Entity Resolution** is an interactive **front-end simulation dashboard** that demonstrates how financial institutions identify **linked entities**, **multi-hop fund flows**, and **potential money-laundering schemes** using graph-based techniques.

This project is built using **pure HTML + CSS + JavaScript**, has **zero backend**, and is ideal for GitHub Pages deployment.

The dashboard simulates:

- Entity resolution in a financial network  
- Multi-hop transaction flows  
- Hazardous clusters (shell companies, offshore accounts, crypto exchanges)  
- GNN-style anomaly scoring  
- Interactive case analysis  
- SAR report export  

This is designed as a **portfolio project + visual demo** of entity resolution concepts—not a production AML engine.

---

## 🎥 Dashboard Preview

<p align="center">
<img src="https://raw.githubusercontent.com/arjunlaxman/aml-entity-resolution/main/assets/preview.png" width="800" alt="Dashboard Preview">
</p>

---

## ✨ Features

### 🧩 1. Entity Resolution Simulation
The system generates a mock set of **accounts, shell companies, offshore holdings, and crypto exchanges**, each with:

- Risk scores  
- Entity categories  
- Relationship edges (transactions)  
- Flags for suspicious behavior  

### 🔗 2. Multi-Hop Transaction Path Analysis
The dashboard simulates:

- Layering schemes  
- Smurfing chains  
- Round-tripping loops  
- Cross-border transfers  
- Crypto aggregation pathways  

### ⚠️ 3. Interactive Case Cards
Each resolved case shows:

- Type of AML pattern  
- Severity level (Critical / High / Medium)  
- Entities involved  
- Total funds moved  
- Risk score and hop length  
- Descriptions of behavior  

### 📊 4. Detailed Entity & Indicator Breakdown
Upon selecting a case, the dashboard displays:

- Entity list with types & risk percentages  
- Behavioral indicators (decay patterns, time windows, shell intermediaries)  
- Network characteristics  

### 📝 5. Exportable SAR Reports
One-click generation of a downloadable `.txt` SAR report containing:

- Case ID  
- Risk score  
- Layering/hop details  
- Indicator list  
- All involved entities  
- Timestamp  

### 🎨 6. Cybersecurity-Themed UI
- Glassmorphism  
- Neon gradients  
- Pulse animations on critical alerts  
- Responsive grid-based layout  
- Smooth hover transitions  
- Custom scrollbars  

---

## 🛠️ Tech Stack

| Layer        | Technology                  |
|-------------|------------------------------|
| Markup      | HTML5                       |
| Styling     | CSS3 (modern, custom styles) |
| Logic       | Vanilla JavaScript (ES6+)    |
| Hosting     | GitHub Pages                 |
| Backend     | None — fully client-side     |

This makes the project extremely lightweight and perfect for portfolios.

---

## 📦 Project Structure

```text
/
├── index.html          # Main dashboard (self-contained UI)
├── README.md           # Documentation
├── assets/             # Demo images, icons, gifs
└── LICENSE             # MIT
