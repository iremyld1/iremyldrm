// Yıl güncelle
document.getElementById('year').textContent = new Date().getFullYear();

// Smooth scroll
document.querySelectorAll('a[href^="#"]').forEach(a=>{
  a.addEventListener('click', e=>{
    e.preventDefault();
    const id = a.getAttribute('href');
    if(id === '#') return;
    const el = document.querySelector(id);
    if(el) el.scrollIntoView({behavior:'smooth'});
  });
});
