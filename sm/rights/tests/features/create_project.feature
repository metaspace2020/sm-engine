Feature: user creates a project

  As a platform user
  I want to be able to create and manage projects
  So that I can collaborate with people outside my research group

  Background:
    Given existing projects
      | name       |
      | Project A  |
      | Project B  |
      | Project C  |

  Scenario Outline: user chooses an existing project name
    Given the name <name> has already been taken
    When I try to create a project named <name>
    Then I get an error message
    Examples:
      | name      |
      | Project A |
      | Project C |

  Scenario Outline: user chooses a non-existing project name
    Given the name <name> has not yet been taken
    When I try to create a project named <name>
    Then the project is created
    And I become that project's administrator
    Examples:
      | name      |
      | Project K |
      | Project L |
