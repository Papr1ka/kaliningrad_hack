<template>
  <div class="container" style="max-width: 800px">
    <CreateVacancyComponent
      style="margin-top: 2em"
      @vacancy_created="add_vacancy"
    ></CreateVacancyComponent>

    <div class="input-group mb-3" style="margin-top: 2em">
      <input type="text" class="form-control" placeholder="Поиск" name="filter" />
      <div class="input-group-append">
        <button @click="searchByFilter" class="btn btn-outline-secondary" type="button">
          Найти
        </button>
      </div>
    </div>

    <div class="row" v-for="vacancy of vacancies" :key="vacancy.name">
      <h6 class="col-4">
        <p>Профессия: {{ vacancy.name }}</p>
        <p>MBTI: {{ vacancy.nbti }}</p>
        <p>HOLLAND: {{ vacancy.holland }}</p>
      </h6>
      <img
        class="col-2"
        width="128px"
        height="128px"
        :src="'http://213.171.28.36:8000/static/assets/' + vacancy.verdict + '.svg'"
      />
      <PlotlyChart
        class="col-6"
        style="min-width: 250px; min-height: 250px"
        :data="vacancy.plot.data"
        :layout="vacancy.plot.layout"
      ></PlotlyChart>
      <div v-if="vacancy.users.length > 0" class="table-responsive">
        <table class="table table-striped">
          <thead>
            <tr>
              <th scope="col">Пользователь</th>
              <th scope="col">Степень подходящести</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(user, index) in vacancy.users" :key="user.name">
              <th scope="row">{{ index }}</th>
              <td>{{ user.name }}</td>
              <td>{{ user.probability }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div v-else>
        <h5>Нет подходящие вариантов</h5>
      </div>
      <hr />
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
import CreateVacancyComponent from '../components/CreateVacancyComponent.vue'
import PlotlyChart from '../components/PlotlyChart.vue'
import { getAllVacancies } from '@/api'

export default {
  components: { PlotlyChart, CreateVacancyComponent },
  mounted() {
    setTimeout(async () => {
      await this.getAll(0)
    }, 0)
  },

  data() {
    return {
      filter: '',
      page: 0,
      count: 1000,
      vacancies: [
        {
          name: '',
          plot: {
            data: [],
            layout: {},
          },
          users: [
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
    add_vacancy(data) {
      this.vacancies.unshift(data)
    },

    async getAll(page) {
      let data = await getAllVacancies(page * 10, 10, this.filter)

      let vacancies = data.data
      let count = data.count
      this.vacancies = vacancies
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
