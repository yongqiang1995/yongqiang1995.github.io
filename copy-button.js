<!-- _includes/footer.html -->
<script>
document.addEventListener('DOMContentLoaded', function() {
  // 动态调整按钮尺寸的函数
  function adjustCopyButtonSize() {
    document.querySelectorAll('.copy-code-btn').forEach(btn => {
      const codeBlock = btn.closest('.highlighter-rouge');
      if (!codeBlock) return;
      
      const codeBlockHeight = codeBlock.offsetHeight;
      // 动态计算尺寸（示例比例）
      btn.style.width = `${Math.min(60, codeBlockHeight * 0.3)}px`;
      btn.style.height = `${Math.max(20, codeBlockHeight * 0.15)}px`;
    });
  }

  // 初始调整
  adjustCopyButtonSize();
  
  // 监听窗口大小变化（可选）
  window.addEventListener('resize', adjustCopyButtonSize);
});
</script>

function debounce(func, timeout = 100) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => func.apply(this, args), timeout);
  };
}

// 修改监听器
window.addEventListener('resize', debounce(adjustCopyButtonSize));