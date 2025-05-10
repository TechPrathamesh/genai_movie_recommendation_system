// Handle rating form submission
document.addEventListener('DOMContentLoaded', function() {
    const ratingForm = document.getElementById('rating-form');
    if (ratingForm) {
        ratingForm.addEventListener('submit', function(e) {
            const rating = document.getElementById('rating').value;
            const watchDuration = document.getElementById('watch-duration').value;
            
            if (rating < 0 || rating > 5) {
                e.preventDefault();
                alert('Rating must be between 0 and 5');
                return;
            }
            
            if (watchDuration < 0) {
                e.preventDefault();
                alert('Watch duration cannot be negative');
                return;
            }
        });
    }
});

// Handle flash messages
document.addEventListener('DOMContentLoaded', function() {
    const flashMessages = document.querySelectorAll('.flash-message');
    flashMessages.forEach(function(message) {
        setTimeout(function() {
            message.style.opacity = '0';
            setTimeout(function() {
                message.remove();
            }, 300);
        }, 3000);
    });
});

// Handle movie card hover effects
document.addEventListener('DOMContentLoaded', function() {
    const movieCards = document.querySelectorAll('.movie-card');
    movieCards.forEach(function(card) {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
            this.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.2)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
            this.style.boxShadow = '0 2px 4px rgba(0, 0, 0, 0.1)';
        });
    });
});

// Handle star rating display
function displayStarRating(rating) {
    const stars = '★'.repeat(Math.floor(rating)) + '☆'.repeat(5 - Math.floor(rating));
    return stars;
}

// Handle responsive navigation menu
document.addEventListener('DOMContentLoaded', function() {
    const navToggle = document.querySelector('.nav-toggle');
    const navMenu = document.querySelector('.navbar');
    
    if (navToggle && navMenu) {
        navToggle.addEventListener('click', function() {
            navMenu.classList.toggle('show');
        });
    }
}); 