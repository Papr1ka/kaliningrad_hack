<template>
  <div class="container" style="max-width: 800px">
    <CreateVacancyComponent
      @user_created="add_user"
      style="margin-top: 2em"
    ></CreateVacancyComponent>

    <div class="input-group mb-3" style="margin-top: 2em">
      <input type="text" class="form-control" placeholder="Поиск" name="filter" />
      <div class="input-group-append">
        <button @click="searchByFilter" class="btn btn-outline-secondary" type="button">
          Найти
        </button>
      </div>
    </div>

    <div class="row" v-for="user of users" :key="user.name">
      <h6 class="col-4">
        <p>Пользователь: {{ user.name }}</p>
        <p>MBTI: {{ user.nbti }}</p>
        <p>HOLLAND: {{ user.holland }}</p>
      </h6>
      <img
        class="col-2"
        width="128px"
        height="128px"
        :src="'http://213.171.28.36:8000/static/assets/' + user.verdict + '.svg'"
      />
      <PlotlyChart
        style="min-width: 250px; min-height: 250px"
        class="col-6"
        :data="user.plot.data"
        :layout="user.plot.layout"
      ></PlotlyChart>
      <div v-if="user.vacancies.length > 0" class="table-responsive">
        <table class="table table-striped">
          <thead>
            <tr>
              <th scope="col">Вакансия</th>
              <th scope="col">Степень подходящести</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(vacancy, index) in user.vacancies" :key="vacancy.name">
              <th scope="row">{{ index }}</th>
              <td>{{ vacancy.name }}</td>
              <td>{{ vacancy.probability }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div v-else>
        <h5>Нет подходящих вариантов</h5>
      </div>
    </div>
    <div
      style="display: flex; align-items: center; justify-content: center; flex-direction: column"
    >
      <nav aria-label="Page navigation example">
        <ul class="pagination">
          <li class="page-item" v-if="this.page != 0">
            <a @click="this.page = 0" class="page-link" aria-label="Previous">
              <span aria-hidden="true">&laquo;</span>
              <span class="sr-only">{{ 1 }}</span>
            </a>
          </li>
          <li class="page-item" v-if="this.page != 0">
            <a @click="this.page -= 1" class="page-link">{{ page - 1 }}</a>
          </li>
          <li class="page-item">
            <a class="page-link active">{{ page }}</a>
          </li>
          <li class="page-item" v-if="hasNext && this.page != lastPage">
            <a @click="this.page += 1" class="page-link">{{ page + 1 }}</a>
          </li>
          <li class="page-item" v-if="hasNext && this.page != lastPage">
            <a @click="this.page = this.lastPage" class="page-link" aria-label="Next">
              <span aria-hidden="true">&raquo;</span>
              <span class="sr-only">{{ lastPage + 1 }}</span>
            </a>
          </li>
        </ul>
      </nav>
    </div>
  </div>
</template>

<script>
import CreateVacancyComponent from '../components/CreateUserComponent.vue'
import PlotlyChart from '../components/PlotlyChart.vue'
import { getAllUsers } from '@/api'

export default {
  components: { PlotlyChart, CreateVacancyComponent },
  mounted() {
    setTimeout(this.getAll(0), 0)
  },

  data() {
    return {
      filter: '',
      page: 0,
      count: 1000,
      users: [
        {
          name: '',
          plot: {
            data: [],
            layout: {},
          },
          vacancies: [
            {
              name: '',
              probability: '',
            },
          ],
        },
      ],
    }
  },

  computed: {
    hasNext() {
      return this.page * 10 < this.count
    },

    lastPage() {
      return Math.ceil(this.count / 10)
    },
  },

  methods: {
    add_user(data) {
      this.users.unshift(data)
    },

    async getAll(page) {
      let data = await getAllUsers(page * 10, 10, this.filter)

      let users = data.data
      let count = data.count
      this.users = users
      this.count = count
    },

    async searchByFilter() {
      this.getAll(0)
    },
  },

  watch: {
    page(new_value) {
      console.log(new_value)
      this.getAll(new_value)
    },
  },
}
</script>

<style></style>
