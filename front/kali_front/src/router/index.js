import { createRouter, createWebHistory } from 'vue-router'
import VacanciesView from '../views/VacanciesView.vue'
import UsersComponent from '@/views/UsersComponent.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: UsersComponent,
    },
    {
      path: '/vacancies',
      name: 'vacancies',
      component: VacanciesView,
    },
  ],
})

export default router
