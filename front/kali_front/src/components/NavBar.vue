<template>
  <nav class="navbar navbar-expand-lg navbar-light bg-light">
    <div class="container">
      <div class="navbar-collapse justify-content-center" id="navbarNav">
        <ul class="navbar-nav">
          <li class="nav-item">
            <RouterLink class="nav-link" to="/">Пользователи</RouterLink>
          </li>
          <li class="nav-item">
            <RouterLink class="nav-link" to="/vacancies">Вакансии</RouterLink>
          </li>
        </ul>
      </div>
    </div>
  </nav>
</template>

<script>
export default {
  name: 'NavBar',
}
</script>

<style scoped>
.navbar {
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.nav-link {
  padding: 1rem 1.5rem;
  font-weight: 500;
}

.nav-link:hover {
  color: #ff7518;
}
</style>
