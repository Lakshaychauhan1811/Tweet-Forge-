

class TweetGenerator {
    constructor() {
        this.lastFormData = null;
        this.userStatus = { logged_in: false, tier: 'free', max_chars: 280 };
        this.initializeEventListeners();
        this.updateCharacterCount();
        this.initializeAnimations();
        this.initializeParticleEffects();
        this.checkUserStatus();
    }

    initializeEventListeners() {
        const form = document.getElementById('tweetForm');
        const generateBtn = document.getElementById('generateBtn');
        // tweet length segmented removed from UI; ensure hidden value remains
        const maxHidden = document.getElementById('max_chars');
        if (maxHidden && !maxHidden.value) maxHidden.value = '280';
        if (form) {
            form.addEventListener('submit', (e) => this.handleFormSubmit(e));
        }
        if (generateBtn) {
            generateBtn.addEventListener('click', (e) => {
                // Ensure click always triggers submission logic
                e.preventDefault();
                console.log('[TweetForge] Generate button clicked');
                this.handleFormSubmit(e);
            });
        }
        const hashtagsInput = document.getElementById('hashtags');
        if (hashtagsInput) {
            hashtagsInput.addEventListener('input', () => this.updateCharacterCount());
        }
        // No manual suggest button; suggestions appear after generation
        // Add event listener for copy button (support id or class)
        const bindCopy = () => {
            const copyBtn = document.querySelector('.copy-btn') || document.getElementById('copyBtn');
            if (copyBtn && !copyBtn.dataset.bound) {
                copyBtn.addEventListener('click', () => this.copyTweet());
                copyBtn.dataset.bound = 'true';
            }
        };
        bindCopy();
        // Post to X button
        const postBtn = document.getElementById('postToXBtn');
        if (postBtn) {
            postBtn.addEventListener('click', async (e) => {
                e.preventDefault();
                const tweetContent = document.getElementById('tweet-content')?.textContent || '';
                if (!tweetContent) {
                    this.showMessage('No tweet content to post.', 'error');
                    return;
                }
                
                // Show posting indicator
                postBtn.disabled = true;
                postBtn.innerHTML = '<i class="fas fa-spinner fa-spin" style="margin-right: 8px;"></i> Posting...';
                
                try {
                    const fd = new FormData();
                    fd.append('tweet', tweetContent);
                    const resp = await fetch('/post-to-x', { method: 'POST', body: fd });
                    if (!resp.ok) {
                        const err = await resp.json().catch(() => ({}));
                        throw new Error(err.detail || 'Failed to post');
                    }
                    const data = await resp.json();
                    this.showMessage('Tweet posted to X successfully!', 'success');
                    
                    // Add visual confirmation
                    postBtn.classList.add('success');
                    postBtn.innerHTML = '<i class="fas fa-check" style="margin-right: 8px;"></i> Posted Successfully';
                    
                    // Reset button after delay
                    setTimeout(() => {
                        postBtn.classList.remove('success');
                        postBtn.disabled = false;
                        postBtn.innerHTML = '<i class="fab fa-x-twitter" style="margin-right: 8px;"></i> Post to X';
                    }, 3000);
                } catch (err) {
                    this.showMessage(err.message || 'Posting failed', 'error');
                    postBtn.disabled = false;
                    postBtn.innerHTML = '<i class="fab fa-x-twitter" style="margin-right: 8px;"></i> Post to X';
                }
            });
        }

        // YouTube form
        // youtube length segmented removed; ensure hidden value remains
        const ytHidden = document.getElementById('yt_max_chars');
        if (ytHidden && !ytHidden.value) ytHidden.value = '280';
        const ytForm = document.getElementById('ytForm');
        if (ytForm) {
            ytForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                const url = (document.getElementById('yt_url')?.value || '').trim();
                if (!url) return this.showMessage('Enter a YouTube URL', 'error');
                const max = document.getElementById('yt_max_chars')?.value || '280';
                this.showLoading(true);
                try {
                    const fd = new FormData();
                    fd.append('url', url);
                    fd.append('max_chars', max);
                    // enable research to use Serper for latest context
                    fd.append('allow_research', 'true');
                    const resp = await fetch('/generate-tweet-youtube', { method: 'POST', body: fd });
                    if (!resp.ok) throw new Error((await resp.json()).detail || 'Server error');
                    const data = await resp.json();
                    if (data.success) {
                        this.displayTweet(data);
                        this.showMessage('Tweet generated from YouTube!', 'success');
                    } else {
                        this.showMessage('Failed to generate', 'error');
                    }
                } catch (err) {
                    this.showMessage(err.message || 'Error', 'error');
                } finally {
                    this.showLoading(false);
                }
            });
        }

       
    }

    async checkUserStatus() {
        try {
            const response = await fetch('/api/user-status');
            if (response.ok) {
                this.userStatus = await response.json();
                this.updateUIForUserStatus();
            }
        } catch (error) {
            console.log('Could not check user status:', error);
        }
    }

    updateUIForUserStatus() {
        // Update login button text
        const loginBtn = document.querySelector('a[href="/login/x"]');
        if (loginBtn) {
            if (this.userStatus.logged_in) {
                loginBtn.innerHTML = '<i class="fab fa-x-twitter"></i> Logged in (' + this.userStatus.tier + ')';
                loginBtn.href = '/logout/x';
            } else {
                loginBtn.innerHTML = '<i class="fab fa-x-twitter"></i> Login with X';
                loginBtn.href = '/login/x';
            }
        }

        // Update character limits based on user tier
        if (this.userStatus.logged_in) {
            const maxChars = this.userStatus.max_chars;
            // Update hidden inputs only
            const hidden = document.getElementById('max_chars');
            if (hidden) hidden.value = maxChars;
            const yth = document.getElementById('yt_max_chars');
            if (yth) yth.value = maxChars;
        }
    }

    // removed segmented control updater

    // segmented control helpers removed

    updateCharacterCount() {
        const hashtagsInput = document.getElementById('hashtags');
        const counter = document.querySelector('.character-count');
        if (hashtagsInput && counter) {
            const hashtags = hashtagsInput.value.split(',').filter(tag => tag.trim() !== '');
            counter.textContent = `${hashtags.length}/5 hashtags`;
            if (hashtags.length > 5) {
                counter.style.color = '#ef4444';
            } else {
                counter.style.color = '#6b7280';
            }
        }
    }

    async handleFormSubmit(e) {
        e.preventDefault();
        console.log('[TweetForge] Submitting form...');

        // Topic required
        const topicEl = document.getElementById('topic');
        const topic = topicEl ? (topicEl.value || '').trim() : '';
        if (!topic) {
            this.showMessage('Please provide a topic or description.', 'error');
            return;
        }

        // Create FormData object for API request
        const apiFormData = new FormData();
        apiFormData.append('topic', topic);
        apiFormData.append('audience', '');
        // max length from segmented control
        let maxChars = 280;
        const hidden = document.getElementById('max_chars');
        if (hidden && hidden.value) {
            maxChars = parseInt(hidden.value, 10) || 280;
        }
        apiFormData.append('max_chars', String(maxChars));
        // optional doc blended with prompt
        const docOpt = document.getElementById('doc_optional');
        const file = docOpt?.files?.[0];
        if (file) {
            apiFormData.append('file', file);
            apiFormData.append('topic', topic);
        }
        apiFormData.append('tweet_type', 'professional');
        apiFormData.append('hashtags', (document.getElementById('hashtags')?.value || ''));
        // Always allow research so Serper can enrich short inputs
        apiFormData.append('allow_research', 'true');

        this.lastFormData = apiFormData;
        const hasFile = !!file;
        this.submitFormData(apiFormData, false, hasFile);
        console.log('[TweetForge] Form data submitted:', Object.fromEntries(apiFormData.entries()));
    }

    async submitFormData(apiFormData, isRegenerate, hasFile = false) {
        console.log('[TweetForge] Sending POST /generate-tweet-v2');
        this.showLoading(true);
        this.clearMessages();
        const tweetDisplay = document.getElementById('tweet-display');
        if (tweetDisplay) {
            tweetDisplay.style.display = 'none';
            tweetDisplay.classList.remove('visible');
        }
        const progressFill = document.querySelector('.progress-fill');
        if (progressFill) {
            progressFill.style.width = '0%';
            setTimeout(() => {
                progressFill.style.width = '100%';
            }, 100);
        }
        try {
            // Log form data entries for debugging
            console.log('[TweetForge] Form data being sent:', [...apiFormData.entries()].map(entry => {
                if (entry[0] === 'file' && entry[1] instanceof File) {
                    return [entry[0], `File: ${entry[1].name} (${entry[1].type}, ${entry[1].size} bytes)`];
                }
                return entry;
            }));
            const loading = document.getElementById('loadingIndicator');
            if (loading) {
                loading.querySelector('p').textContent = 'Creating promotional content and researching the web...';
            }
            
            const response = await fetch('/generate-tweet-v2', {
                method: 'POST',
                body: apiFormData
            });
            console.log('[TweetForge] Response status:', response.status);
            if (!response.ok) {
                let errorMsg = `Error: ${response.status}`;
                try {
                    const errorData = await response.json();
                    errorMsg = errorData.detail || JSON.stringify(errorData);
                    console.error('[TweetForge] Error details:', errorData);
                } catch (e) {
                    console.error('[TweetForge] Failed to parse error response:', e);
                }
                throw new Error(errorMsg);
            }
            const result = await response.json();
            console.log('[TweetForge] Response JSON:', result);
            if (result.success) {
                this.displayTweet(result);
                this.showMessage('Tweet generated successfully!', 'success');
            } else {
                this.showMessage(result.detail || 'Failed to generate tweet', 'error');
            }
        } catch (error) {
            console.error('[TweetForge] Error generating tweet:', error);
            let msg = error?.message || error?.detail || JSON.stringify(error);
            this.showMessage(`An error occurred: ${msg}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    async suggestHashtags() {
        const topicEl = document.getElementById('topic');
        const container = document.getElementById('suggested-hashtags');
        if (!topicEl || !topicEl.value.trim()) {
            this.showMessage('Enter a topic or YouTube URL first.', 'error');
            return;
        }
        try {
            const resp = await fetch('/suggest_hashtags', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: topicEl.value.trim(), max_hashtags: 5 })
            });
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                throw new Error(err.detail || 'Server error');
            }
            const data = await resp.json();
            if (container) {
                container.innerHTML = '';
                data.hashtags.forEach(tag => {
                    const btn = document.createElement('button');
                    btn.className = 'tag-btn';
                    btn.textContent = tag;
                    btn.addEventListener('click', () => {
                        const input = document.getElementById('hashtags');
                        if (!input) return;
                        const existing = (input.value || '').split(',').map(s => s.trim()).filter(Boolean);
                        if (!existing.includes(tag)) existing.push(tag);
                        input.value = existing.join(', ');
                        this.updateCharacterCount();
                    });
                    container.appendChild(btn);
                });
            }
        } catch (e) {
            this.showMessage(e.message || 'Failed to suggest hashtags', 'error');
        }
    }

    showLoading(show) {
        const loading = document.getElementById('loadingIndicator');
        const generateBtn = document.getElementById('generateBtn');
        
        if (loading) {
            loading.style.display = show ? 'block' : 'none';
        }
        
        if (generateBtn) {
            generateBtn.disabled = show;
            generateBtn.textContent = show ? 'Generating...' : 'Generate Tweet';
        }
    }

    displayTweet(data) {
        // Get tweet display elements
        const tweetDisplay = document.getElementById('tweet-display');
        const tweetContent = document.getElementById('tweet-content');
        const tweetTypeBadge = document.getElementById('tweet-type-badge');
        const hashtagsDisplay = document.getElementById('hashtags-display');
        const sourcesDisplay = document.getElementById('sources-display');
        const toneDisplay = document.getElementById('tone-display');
        // Set tweet content with typewriter effect
        if (tweetContent) {
            this.typewriterEffect(tweetContent, data.content);
        }
        // Hide tweet type badge (no 'Simple' or 'Professional' shown)
        if (tweetTypeBadge) {
            tweetTypeBadge.style.display = 'none';
        }
        // Set hashtags
        if (hashtagsDisplay) {
            if (data.hashtags && data.hashtags.length > 0) {
                hashtagsDisplay.innerHTML = '';
                data.hashtags.forEach(hashtag => {
                    const hashtagElement = document.createElement('span');
                    hashtagElement.classList.add('hashtag');
                    hashtagElement.textContent = hashtag;
                    hashtagsDisplay.appendChild(hashtagElement);
                });
            } else {
                hashtagsDisplay.innerHTML = '<span class="no-data">No hashtags</span>';
            }
        }
        // Set research sources
        if (sourcesDisplay) {
            if (data.research_sources && data.research_sources.length > 0) {
                sourcesDisplay.innerHTML = '';
                data.research_sources.forEach(source => {
                    const sourceElement = document.createElement('span');
                    sourceElement.classList.add('source');
                    sourceElement.textContent = source;
                    sourcesDisplay.appendChild(sourceElement);
                });
            } else {
                sourcesDisplay.innerHTML = '<span class="no-data">No sources</span>';
            }
        }
        // Set tone
        if (toneDisplay) {
            if (data.tone) {
                toneDisplay.textContent = data.tone;
            } else {
                toneDisplay.textContent = 'Not specified';
            }
        }
        // Show tweet display with enhanced animation
        if (tweetDisplay) {
            tweetDisplay.style.display = 'block';
            tweetDisplay.style.opacity = '0';
            tweetDisplay.style.transform = 'translateY(50px) scale(0.95)';
            
            setTimeout(() => {
                tweetDisplay.style.transition = 'all 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)';
                tweetDisplay.style.opacity = '1';
                tweetDisplay.style.transform = 'translateY(0) scale(1)';
                tweetDisplay.classList.add('visible');
                
                // Add glow effect
                tweetDisplay.style.boxShadow = '0 30px 60px rgba(0, 0, 0, 0.6), 0 0 30px rgba(0, 212, 255, 0.3)';
                
                tweetDisplay.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }, 100);
        }
        // Regenerate functionality removed
        // Make suggested hashtags clickable to add into input
        const addToInput = (tag) => {
            const input = document.getElementById('hashtags');
            if (!input) return;
            const existing = (input.value || '').split(',').map(s => s.trim()).filter(Boolean);
            if (!existing.includes(tag)) existing.push(tag);
            input.value = existing.join(', ');
            this.updateCharacterCount();
        };
        const tags = document.querySelectorAll('#hashtags-display .hashtag');
        tags.forEach(el => {
            if (!el.dataset.bound) {
                el.addEventListener('click', () => addToInput(el.textContent || ''));
                el.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter' || e.key === ' ') addToInput(el.textContent || '');
                });
                el.dataset.bound = 'true';
                el.style.cursor = 'pointer';
                el.title = 'Click to add hashtag';
                el.setAttribute('role', 'button');
                el.setAttribute('tabindex', '0');
            }
        });
        // Rebind copy after DOM updates
        const copyBtn = document.querySelector('.copy-btn') || document.getElementById('copyBtn');
        if (copyBtn && !copyBtn.dataset.bound) {
            copyBtn.addEventListener('click', () => this.copyTweet());
            copyBtn.dataset.bound = 'true';
        }
        // Clear form fields after tweet is generated
        const form = document.getElementById('tweetForm');
        if (form) {
            form.reset();
            // Also clear file upload label/info
            const fileLabel = document.querySelector('.file-upload-label');
            const fileInfo = document.querySelector('.file-upload-info');
            if (fileLabel) {
                fileLabel.innerHTML = '<i class="fas fa-upload"></i> Drag and drop or click to upload';
                fileLabel.classList.remove('file-selected');
            }
            if (fileInfo) {
                fileInfo.textContent = 'Supports PNG, JPG, JPEG, WEBP, BMP files';
            }
            // Reset hashtag counter
            const counter = document.querySelector('.character-count');
            if (counter) counter.textContent = '0/5 hashtags';
        }
        // Show success message with enhanced styling
        this.showMessage('✨ Tweet generated successfully!', 'success');
    }

    showMessage(message, type = 'info') {
        const container = document.getElementById('message-container');
        if (!container) return;

        const messageElement = document.createElement('div');
        messageElement.className = `message ${type}`;
        
        // Add icon based on message type
        let icon = '';
        if (type === 'success') {
            icon = '<i class="fas fa-check-circle"></i> ';
        } else if (type === 'error') {
            icon = '<i class="fas fa-exclamation-triangle"></i> ';
        } else {
            icon = '<i class="fas fa-info-circle"></i> ';
        }
        
        messageElement.innerHTML = icon + message;
        container.appendChild(messageElement);
        
        // Add entrance animation
        setTimeout(() => {
            messageElement.classList.add('show');
        }, 10);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            messageElement.classList.add('hide');
            setTimeout(() => {
                if (messageElement.parentNode) {
                    messageElement.remove();
                }
            }, 300);
        }, 5000);
    }

    clearMessages() {
        const container = document.getElementById('message-container');
        if (container) {
            container.innerHTML = '';
        }
    }

    // Copy tweet functionality with enhanced animations
    copyTweet() {
        const tweetContent = document.getElementById('tweet-content');
        const hashtagsDisplay = document.getElementById('hashtags-display');
        
        if (!tweetContent) {
            this.showMessage('No tweet content to copy', 'error');
            return;
        }
        
        // Get tweet content
        let content = tweetContent.textContent;
        
        // Get hashtags if available
        if (hashtagsDisplay) {
            const hashtags = Array.from(hashtagsDisplay.querySelectorAll('.hashtag'))
                .map(el => el.textContent)
                .join(' ');
            
            if (hashtags) {
                content += '\n\n' + hashtags;
            }
        }
        
        // Copy to clipboard
        navigator.clipboard.writeText(content)
            .then(() => {
                this.showMessage('Tweet copied to clipboard!', 'success');
                
                // Enhanced visual feedback on the copy button
                const copyBtn = document.querySelector('.copy-btn');
                if (copyBtn) {
                    copyBtn.classList.add('copied');
                    copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
                    
                    // Add success animation
                    copyBtn.style.transform = 'scale(1.05)';
                    copyBtn.style.boxShadow = '0 0 30px rgba(0, 191, 255, 0.6)';
                    
                    setTimeout(() => {
                        copyBtn.classList.remove('copied');
                        copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy Tweet';
                        copyBtn.style.transform = '';
                        copyBtn.style.boxShadow = '';
                    }, 2000);
                }
            })
            .catch(err => {
                console.error('Failed to copy: ', err);
                this.showMessage('Failed to copy tweet', 'error');
            });
    }

    // Initialize enhanced animations
    initializeAnimations() {
        // Add intersection observer for scroll animations
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('fade-in-up');
                    
                    // Add staggered animation for feature cards
                    if (entry.target.classList.contains('feature-card')) {
                        const cards = document.querySelectorAll('.feature-card');
                        const index = Array.from(cards).indexOf(entry.target);
                        entry.target.style.animationDelay = `${index * 0.1}s`;
                    }
                }
            });
        }, observerOptions);

        // Observe elements for animations
        document.querySelectorAll('.feature-card, .form-container, .tweet-display, .privacy-card, .stat-card').forEach(el => {
            observer.observe(el);
        });
    }

    // Initialize particle effects
    initializeParticleEffects() {
        // Add floating particles to the background
        this.createFloatingParticles();
        
        // Add hover effects to interactive elements
        this.addHoverEffects();
    }

    // Create floating particles
    createFloatingParticles() {
        const particleContainer = document.createElement('div');
        particleContainer.className = 'particle-container';
        particleContainer.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: -1;
        `;
        document.body.appendChild(particleContainer);

        // Create particles
        for (let i = 0; i < 30; i++) {
            const particle = document.createElement('div');
            particle.className = 'floating-particle';
            const colors = [
                'linear-gradient(135deg, #00ffff, #9d4edd)',
                'linear-gradient(135deg, #ff006e, #ff4500)',
                'linear-gradient(135deg, #39ff14, #00ffff)',
                'linear-gradient(135deg, #ffff00, #ff006e)'
            ];
            const randomColor = colors[Math.floor(Math.random() * colors.length)];
            const size = 3 + Math.random() * 4; // 3-7px
            
            particle.style.cssText = `
                position: absolute;
                width: ${size}px;
                height: ${size}px;
                background: ${randomColor};
                border-radius: 50%;
                opacity: 0.6;
                animation: float ${5 + Math.random() * 10}s linear infinite;
                left: ${Math.random() * 100}%;
                top: ${Math.random() * 100}%;
                animation-delay: ${Math.random() * 5}s;
                box-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
            `;
            particleContainer.appendChild(particle);
        }

        // Add CSS animation for particles
        const style = document.createElement('style');
        style.textContent = `
            @keyframes float {
                0% { transform: translateY(100vh) rotate(0deg); opacity: 0; }
                10% { opacity: 0.3; }
                90% { opacity: 0.3; }
                100% { transform: translateY(-100px) rotate(360deg); opacity: 0; }
            }
        `;
        document.head.appendChild(style);
    }

    // Add enhanced hover effects
    addHoverEffects() {
        // Add ripple effect to buttons
        document.querySelectorAll('.btn, .cyber-btn, .copy-btn').forEach(button => {
            button.addEventListener('click', (e) => {
                const ripple = document.createElement('span');
                const rect = button.getBoundingClientRect();
                const size = Math.max(rect.width, rect.height);
                const x = e.clientX - rect.left - size / 2;
                const y = e.clientY - rect.top - size / 2;
                
                ripple.style.cssText = `
                    position: absolute;
                    width: ${size}px;
                    height: ${size}px;
                    left: ${x}px;
                    top: ${y}px;
                    background: rgba(255, 255, 255, 0.3);
                    border-radius: 50%;
                    transform: scale(0);
                    animation: ripple 0.6s linear;
                    pointer-events: none;
                `;
                
                button.style.position = 'relative';
                button.style.overflow = 'hidden';
                button.appendChild(ripple);
                
                setTimeout(() => ripple.remove(), 600);
            });
        });

        // Add ripple animation CSS
        const style = document.createElement('style');
        style.textContent = `
            @keyframes ripple {
                to {
                    transform: scale(4);
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(style);
    }

    // Typewriter effect for text
    typewriterEffect(element, text, speed = 30) {
        element.textContent = '';
        let i = 0;
        const timer = setInterval(() => {
            if (i < text.length) {
                element.textContent += text.charAt(i);
                i++;
            } else {
                clearInterval(timer);
            }
        }, speed);
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const tweetGenerator = new TweetGenerator();
    
    // Initialize progress animation
    const progressFill = document.querySelector('.progress-fill');
    if (progressFill) {
        progressFill.style.width = '0%';
    }
    
    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

});

// Add fade-in animation for elements
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('fade-in-up');
        }
    });
}, observerOptions);

// Observe all feature cards and other elements
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.feature-card, .form-container, .tweet-display, .privacy-card').forEach(el => {
        observer.observe(el);
    });
});

// Stats counter animation
function animateCounters() {
  const counters = document.querySelectorAll('.stat-number[data-counter]');
  const observer = new IntersectionObserver((entries, obs) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const el = entry.target;
        const target = parseInt(el.getAttribute('data-counter') || '0', 10);
        const duration = 1200; // ms
        const start = performance.now();
        const step = (now) => {
          const p = Math.min(1, (now - start) / duration);
          const val = Math.floor(target * p);
          el.textContent = `${val}%`;
          if (p < 1) requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
        obs.unobserve(el);
      }
    });
  }, { threshold: .4 });
  counters.forEach(c => observer.observe(c));
}

document.addEventListener('DOMContentLoaded', animateCounters);

async function checkLogin() {
    const res = await fetch("/api/user-status");
    const data = await res.json();
    if (data.logged_in) {
        document.querySelector(".nav-links").innerHTML = `
            <a href="/dashboard" class="nav-link">Creator</a>
            <a href="/logout" class="nav-link">Logout</a>
        `;
    }
}
checkLogin();
